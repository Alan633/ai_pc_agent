from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections import deque
from threading import Event
from dataclasses import dataclass
from pathlib import Path

from agent.session import AgentSession
from game_baque.capture import (
    CapturedFrame,
    assert_game_foreground,
    capture_game_window,
    capture_game_window_if_swap,
    click_screen_native,
    current_game_action_state,
    focus_game_window,
    probe_game_turn,
)
from game_baque.ev import ExpectedValueEngine
from game_baque.event_vision import TableObservation
from game_baque.fast_vision import analyze_image_path
from game_baque.local_decision import LocalDecisionReranker
from game_baque.models import Card, GameDecision, GameSnapshot, Point, ScreenType
from game_baque.strategy import choose_discard, choose_draw, choose_swap_two
from game_baque.tracker import TableTracker
from game_baque.vision import BaqueVision

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GameStepResult:
    acted: bool
    message: str
    snapshot: GameSnapshot
    frame: CapturedFrame
    stop_loop: bool = False


class StaleActionError(RuntimeError):
    """模型返回后画面已换回合，当前动作必须丢弃。"""


class BaqueGameController:
    def __init__(
        self,
        *,
        vision: BaqueVision,
        screenshots_dir: Path,
        session: AgentSession,
        window_title_keyword: str = "欢乐斗地主",
        action_interval_seconds: float = 1.5,
        min_card_confidence: float = 0.72,
        min_event_confidence: float = 0.80,
        auto_next_round: bool = False,
        decision_budget_seconds: float = 4.0,
        fast_vision_enabled: bool = True,
        decision_mode: str = "ev",
        decision_reranker: LocalDecisionReranker | None = None,
    ) -> None:
        self.vision = vision
        self.screenshots_dir = screenshots_dir
        self.session = session
        self.window_title_keyword = window_title_keyword
        self.action_interval_seconds = max(0.5, float(action_interval_seconds))
        self.min_card_confidence = min(max(float(min_card_confidence), 0.0), 1.0)
        self.min_event_confidence = min(max(float(min_event_confidence), 0.0), 1.0)
        self.auto_next_round = bool(auto_next_round)
        self.fast_vision_enabled = bool(fast_vision_enabled)
        normalized_mode = decision_mode.strip().lower()
        if normalized_mode not in {"rule", "ev", "hybrid"}:
            raise ValueError("decision_mode 必须是 rule、ev 或 hybrid")
        self.decision_mode = normalized_mode
        self.decision_reranker = decision_reranker
        # 回合约 20 秒，识别占 4~6 秒；EV 决策必须在预算内给出结果。
        self.decision_budget_seconds = max(0.5, float(decision_budget_seconds))
        self.last_snapshot: GameSnapshot | None = None
        self.last_message = "尚未识别"
        self.action_count = 0
        self.tracker = TableTracker()
        self.ev = ExpectedValueEngine()
        self._tracker_started = False
        self._round_gold_ranks: tuple[str, ...] = ()
        self._last_action_fingerprint: tuple[object, ...] | None = None
        self._last_action_at = 0.0
        self._observation_dedupe: deque[tuple[float, tuple[object, ...]]] = deque(maxlen=24)
        self._observer_recent_events: deque[str] = deque(maxlen=12)
        self._previous_frame_path: Path | None = None
        self._pair_event_dedupe: dict[tuple[str, str, str], float] = {}
        self.external_observer_active = False
        self.opponent_event_count = 0
        self.inspect_count = 0
        self.unknown_snapshot_count = 0
        self.step_count = 0
        self.validation_rejection_count = 0
        self.fast_ui_count = 0
        self.qwen_vision_count = 0
        self.fast_ui_fallback_count = 0
        self.hybrid_decision_count = 0
        self.hybrid_fallback_count = 0
        self.last_decision_source = "none"

    async def inspect(
        self,
        *,
        include_previous: bool = True,
    ) -> tuple[CapturedFrame, GameSnapshot]:
        frame = await asyncio.to_thread(
            capture_game_window,
            self.screenshots_dir,
            self.window_title_keyword,
        )
        snapshot = await self._parse_captured_frame(frame, include_previous=include_previous)
        return frame, snapshot

    async def _parse_captured_frame(
        self,
        frame: CapturedFrame,
        *,
        include_previous: bool,
    ) -> GameSnapshot:
        """固定 UI 先走 OpenCV；只有牌面/异常页面才调用 Qwen。"""
        fast = (
            await asyncio.to_thread(analyze_image_path, frame.path)
            if self.fast_vision_enabled
            else None
        )

        # 开始页完全不需要多模态模型：OpenCV 直接给出按钮中心。
        if (
            fast is not None
            and fast.screen_type == ScreenType.START
            and fast.start_button is not None
            and fast.confidence >= 0.72
        ):
            self.fast_ui_count += 1
            snapshot = GameSnapshot(
                screen_type=ScreenType.START,
                start_button=fast.start_button,
                message=f"{fast.message} confidence={fast.confidence:.2f}",
            )
            LOGGER.info(
                "八雀OpenCV命中 start confidence=%.2f x=%s y=%s",
                fast.confidence,
                fast.start_button.x,
                fast.start_button.y,
            )
            self._register_snapshot(frame, snapshot)
            return snapshot

        # 启动流程或外部观察器运行时，不再用整图 Qwen 判断“是不是轮到我”。
        # OpenCV 只判断阶段；真正牌面仍交给专用快速视觉协议。
        can_use_fast_turn = not include_previous or self.external_observer_active
        if fast is not None and fast.action_state is not None and can_use_fast_turn:
            self.fast_ui_count += 1
            self.qwen_vision_count += 1
            LOGGER.info(
                "八雀OpenCV命中 action=%s，Qwen只识别牌面",
                fast.action_state,
            )
            snapshot = await self.vision.parse_turn(frame.path, fast.action_state)
            self._register_snapshot(frame, snapshot)
            return snapshot

        # 牌局进行中但不是自己回合时，自动模式无需浪费数秒让 Qwen重复确认。
        if fast is not None and fast.round_in_progress and can_use_fast_turn:
            self.fast_ui_count += 1
            snapshot = GameSnapshot(
                screen_type=ScreenType.WAITING,
                message=fast.message,
            )
            self._register_snapshot(frame, snapshot)
            return snapshot

        # 结算、充值、陌生弹窗，以及需要用两帧补对手事件的手动模式仍交给 Qwen。
        self.fast_ui_fallback_count += 1
        self.qwen_vision_count += 1
        LOGGER.info("八雀OpenCV未覆盖当前页面，回退Qwen视觉识别")
        snapshot = await self.vision.parse(
            frame.path,
            previous_image_path=(
                None
                if self.external_observer_active or not include_previous
                else self._previous_frame_path
            ),
        )
        self._register_snapshot(frame, snapshot)
        return snapshot

    def _register_snapshot(self, frame: CapturedFrame, snapshot: GameSnapshot) -> None:
        self._previous_frame_path = frame.path
        self._sync_tracker(snapshot)
        if not self.external_observer_active:
            self._apply_opponent_events(snapshot)
        # 当前截图里的 deck_count 是视觉观察到的最终状态；
        # 对手暗摸事件会先按事件语义减 1，这里再用观察值校准，避免双重扣减。
        if snapshot.deck_count is not None and self._tracker_started:
            self.tracker.set_deck_count(snapshot.deck_count)
        self.inspect_count += 1
        if snapshot.screen_type == ScreenType.UNKNOWN:
            self.unknown_snapshot_count += 1
        self.last_snapshot = snapshot
        self.last_message = self._snapshot_summary(snapshot)

    async def _wait_for_turn(
        self,
        stop_signal: Event,
        *,
        idle_timeout_seconds: float = 90.0,
    ) -> tuple[CapturedFrame, GameSnapshot] | None:
        """用像素门等待自己的操作回合，只在回合开始时调用视觉模型。

        本地视觉单帧需要数秒，而回合倒计时约 20 秒。若在对手回合持续识别，
        轮到自己时模型往往正忙于旧画面，返回时倒计时已所剩无几。返回 None
        表示牌局不在进行中（结算/弹窗/开始页）或长时间无回合，交给完整识别。
        """
        deadline = time.monotonic() + idle_timeout_seconds
        while not stop_signal.is_set() and time.monotonic() < deadline:
            probe = await asyncio.to_thread(
                probe_game_turn,
                self.screenshots_dir,
                self.window_title_keyword,
            )
            if probe.frame is not None and probe.action_state is not None:
                started_at = time.monotonic()
                try:
                    snapshot = await self.vision.parse_turn(
                        probe.frame.path,
                        probe.action_state,
                    )
                except (RuntimeError, ValueError) as exc:
                    # 单帧 JSON 损坏/空回复不应结束整局，回合内继续重探。
                    LOGGER.warning("八雀回合识别失败，重新探测：%s", exc)
                    continue
                LOGGER.info(
                    "八雀回合识别完成 state=%s duration=%.2fs hand=%s",
                    probe.action_state,
                    time.monotonic() - started_at,
                    len(snapshot.hand),
                )
                self._register_snapshot(probe.frame, snapshot)
                return probe.frame, snapshot
            if not probe.round_in_progress:
                return None
            await asyncio.sleep(0.25)
        return None

    async def _wait_for_swap_frame(self, timeout_seconds: float = 25.0) -> CapturedFrame | None:
        deadline = time.monotonic() + max(1.0, timeout_seconds)
        while time.monotonic() < deadline:
            try:
                frame = await asyncio.to_thread(
                    capture_game_window_if_swap,
                    self.screenshots_dir,
                    self.window_title_keyword,
                )
            except RuntimeError:
                await asyncio.to_thread(focus_game_window, self.window_title_keyword)
                await asyncio.sleep(0.20)
                continue
            if frame is not None:
                LOGGER.info("八雀启动流程：快速检测到换牌界面")
                return frame
            await asyncio.sleep(0.25)
        return None

    async def step(
        self,
        *,
        execute: bool = True,
        prefetched: tuple[CapturedFrame, GameSnapshot] | None = None,
    ) -> GameStepResult:
        self.step_count += 1
        frame, snapshot = prefetched if prefetched is not None else await self.inspect()

        if snapshot.screen_type == ScreenType.START:
            if snapshot.start_button is None:
                return GameStepResult(
                    False,
                    "已识别到八雀开始页面，但没有可靠识别到“开始游戏”按钮，暂不点击。",
                    snapshot,
                    frame,
                )
            if execute:
                await self._click(frame, snapshot.start_button)
                self.action_count += 1
                self._last_action_at = time.monotonic()
                LOGGER.info(
                    "八雀启动流程：已自动点击开始游戏 x=%s y=%s",
                    snapshot.start_button.x,
                    snapshot.start_button.y,
                )
            return GameStepResult(
                execute,
                "已点击“开始游戏”，等待发牌。" if execute else "当前在开始页面，可点击“开始游戏”。",
                snapshot,
                frame,
            )

        if snapshot.screen_type == ScreenType.RECHARGE:
            return GameStepResult(
                False,
                "检测到“豆子不足/充值/购买”界面。自动游戏不会充值或购买，已停止。",
                snapshot,
                frame,
                stop_loop=True,
            )
        if snapshot.screen_type == ScreenType.SETTLEMENT:
            if not self.auto_next_round:
                return GameStepResult(
                    False,
                    "本局已结束。GAME_AUTO_NEXT_ROUND=false，自动停止，避免自动消耗下一局豆子。",
                    snapshot,
                    frame,
                    stop_loop=True,
                )
            if snapshot.continue_button is None:
                return GameStepResult(False, "已到结算页，但未可靠识别继续按钮。", snapshot, frame)
            if execute:
                await self._click(frame, snapshot.continue_button)
                self.action_count += 1
            return GameStepResult(execute, "点击继续进入下一局。", snapshot, frame)

        if snapshot.screen_type in {ScreenType.WAITING, ScreenType.UNKNOWN}:
            return GameStepResult(
                False,
                snapshot.message or f"当前状态：{snapshot.screen_type.value}，等待界面变化。",
                snapshot,
                frame,
            )

        validation_error = self._validate_actionable_snapshot(snapshot, frame)
        if validation_error:
            self.validation_rejection_count += 1
            return GameStepResult(False, validation_error, snapshot, frame)

        required_action = {
            ScreenType.SWAP: "swap",
            ScreenType.DRAW: "draw",
            ScreenType.DISCARD: "discard",
            ScreenType.HU: "discard",
        }.get(snapshot.screen_type)
        if required_action is not None:
            current_action = await self._current_action_state()
            if current_action != required_action:
                return GameStepResult(
                    False,
                    (
                        f"模型识别为 {snapshot.screen_type.value}，但当前即时画面"
                        f"操作权为 {current_action or '对手回合/无'}，丢弃过期动作。"
                    ),
                    snapshot,
                    frame,
                )

        fingerprint = snapshot.fingerprint()
        if (
            execute
            and fingerprint == self._last_action_fingerprint
            and time.monotonic() - self._last_action_at < max(3.0, self.action_interval_seconds * 1.5)
        ):
            return GameStepResult(False, "画面尚未变化，等待游戏响应。", snapshot, frame)

        # EV/蒙特卡洛属于 CPU 密集计算，放到工作线程，避免阻塞 6 FPS 对手观察器。
        # 决策使用当前 TableTracker 的深拷贝，避免后台观察器在计算过程中修改同一牌池。
        tracker_snapshot = copy.deepcopy(self.tracker)
        started_at = time.monotonic()
        decision = await asyncio.to_thread(self._decide, snapshot, tracker_snapshot)
        LOGGER.info(
            "八雀决策完成 state=%s action=%s duration=%.2fs",
            snapshot.screen_type.value,
            decision.action,
            time.monotonic() - started_at,
        )
        if not execute:
            return GameStepResult(False, decision.reason, snapshot, frame)

        try:
            message = await self._execute(frame, snapshot, decision)
        except StaleActionError as exc:
            return GameStepResult(False, str(exc), snapshot, frame)
        self.action_count += 1
        self._last_action_fingerprint = fingerprint
        self._last_action_at = time.monotonic()
        return GameStepResult(True, message, snapshot, frame)

    async def prepare_auto_round(self, timeout_seconds: float = 30.0) -> GameSnapshot:
        """自动模式启动前，把游戏切到前台并从“开始游戏”页面进入正式牌局。

        只有识别到真实“开始游戏”按钮才点击，不使用固定坐标；进入换牌 6 张
        或后续 7/8 张手牌
        和 3 个金牌的正式牌局后才返回，随后插件再启动三家观察器。
        """
        await asyncio.to_thread(focus_game_window, self.window_title_keyword)
        # Windows 前台切换存在极短的异步窗口，给系统一点时间完成激活。
        await asyncio.sleep(0.30)
        LOGGER.info("八雀启动流程：已切换游戏窗口到前台")

        deadline = time.monotonic() + max(5.0, float(timeout_seconds))
        start_click_attempts = 0
        last_start_click_at = 0.0
        prefetched: tuple[CapturedFrame, GameSnapshot] | None = None
        while time.monotonic() < deadline:
            # 自动启动只关心当前处于开始、匹配、发牌还是正式牌局；历史帧和
            # 对手事件此时没有价值，双图反而会让本地视觉模型显著变慢。
            if prefetched is not None:
                frame, snapshot = prefetched
                prefetched = None
            else:
                frame, snapshot = await self.inspect(include_previous=False)

            if snapshot.screen_type == ScreenType.RECHARGE:
                raise RuntimeError("检测到豆子不足/充值/购买界面，自动游戏不会继续。")

            start_button = snapshot.start_button
            if start_button is not None and not (
                frame.width * 0.30 <= start_button.x <= frame.width * 0.70
                and frame.height * 0.50 <= start_button.y <= frame.height * 0.85
            ):
                LOGGER.warning(
                    "忽略不可信开始按钮坐标 x=%s y=%s frame=%sx%s screen=%s",
                    start_button.x,
                    start_button.y,
                    frame.width,
                    frame.height,
                    snapshot.screen_type.value,
                )
                start_button = None
            looks_like_start = (
                start_button is not None
                and snapshot.screen_type
                in {ScreenType.START, ScreenType.WAITING, ScreenType.UNKNOWN}
                and not snapshot.hand
                and not snapshot.gold_ranks
            )
            if (
                snapshot.screen_type == ScreenType.START
                and start_button is None
                and start_click_attempts == 0
            ):
                raise RuntimeError(
                    "已进入八雀开始页面，但没有可靠识别到中央“开始游戏”按钮。"
                )
            if looks_like_start:
                now = time.monotonic()
                if start_click_attempts < 3 and now - last_start_click_at >= 3.0:
                    assert start_button is not None
                    await self._click(frame, start_button)
                    self.action_count += 1
                    self._last_action_at = now
                    last_start_click_at = now
                    start_click_attempts += 1
                    if start_click_attempts == 1:
                        # 本地视觉首次识别可能接近原始 30 秒期限；点击成功后应
                        # 给匹配、发牌和后续识别一个完整等待窗口。
                        deadline = time.monotonic() + max(90.0, float(timeout_seconds))
                    LOGGER.info(
                        "八雀启动流程：自动点击开始游戏 attempt=%s x=%s y=%s screen=%s",
                        start_click_attempts,
                        start_button.x,
                        start_button.y,
                        snapshot.screen_type.value,
                    )
                    if start_click_attempts == 1:
                        # 发牌动画无需调用视觉模型。先用本地像素快速等待换牌按钮，
                        # 一出现就只识别这一帧，最大限度保留换牌倒计时。
                        swap_frame = await self._wait_for_swap_frame()
                        swap_snapshot: GameSnapshot | None = None
                        while swap_frame is not None:
                            started_at = time.monotonic()
                            swap_snapshot = await self.vision.parse_swap(swap_frame.path)
                            LOGGER.info(
                                "八雀启动流程：换牌专用识别完成 duration=%.2fs hand=%s",
                                time.monotonic() - started_at,
                                len(swap_snapshot.hand),
                            )
                            self._previous_frame_path = swap_frame.path
                            self._sync_tracker(swap_snapshot)
                            prefetched = (swap_frame, swap_snapshot)
                            if (
                                len(swap_snapshot.hand) in {6, 7}
                                and len(swap_snapshot.gold_ranks) == 3
                            ):
                                break
                            # 漏读一张牌时，只要换牌按钮还在就用新帧再试一次快速
                            # 识别；完整识别约 13 秒，会直接错过换牌倒计时。
                            LOGGER.warning(
                                "八雀启动换牌识别不完整 hand=%s gold=%s，重新识别",
                                len(swap_snapshot.hand),
                                swap_snapshot.gold_ranks,
                            )
                            prefetched = None
                            swap_frame = await self._wait_for_swap_frame(timeout_seconds=1.0)
                        if prefetched is None and swap_snapshot is not None:
                            # 牌局已开始但换牌没能识别完整：交给按回合门控的主循环，
                            # 不再用慢速完整识别轮询启动流程。
                            LOGGER.warning("八雀启动换牌未完成，转入正式游戏循环")
                            return swap_snapshot
                await asyncio.sleep(0.8)
                continue

            if snapshot.screen_type == ScreenType.SETTLEMENT:
                if self.auto_next_round and snapshot.continue_button is not None:
                    await self._click(frame, snapshot.continue_button)
                    self.action_count += 1
                    LOGGER.info("八雀启动流程：结算页自动点击继续")
                    await asyncio.sleep(0.8)
                    continue
                raise RuntimeError(
                    "当前停在上一局结算页；GAME_AUTO_NEXT_ROUND=false，未自动开始下一局。"
                )

            expected_hand_count = {
                ScreenType.DRAW: 7,
                ScreenType.DISCARD: 8,
                ScreenType.HU: 8,
            }.get(snapshot.screen_type)
            if (
                (
                    (snapshot.screen_type == ScreenType.SWAP and len(snapshot.hand) in {6, 7})
                    or (
                        expected_hand_count is not None
                        and len(snapshot.hand) == expected_hand_count
                    )
                )
                and len(snapshot.gold_ranks) == 3
            ):
                LOGGER.info(
                    "八雀启动流程：正式牌局已就绪 state=%s hand=%s gold=%s deck=%s",
                    snapshot.screen_type.value,
                    len(snapshot.hand),
                    snapshot.gold_ranks,
                    snapshot.deck_count,
                )
                # 换牌倒计时很短。这里已经有可执行的当前帧，若返回后让主循环
                # 再做一次本地视觉识别，往往会直接错过倒计时，因此立即完成
                # “选择两张牌 -> 点击换牌按钮”。
                if snapshot.screen_type == ScreenType.SWAP:
                    validation_error = self._validate_actionable_snapshot(snapshot, frame)
                    if validation_error is not None:
                        LOGGER.warning("八雀启动换牌暂不操作：%s", validation_error)
                    else:
                        decision = await asyncio.to_thread(
                            self._decide,
                            snapshot,
                            copy.deepcopy(self.tracker),
                        )
                        try:
                            message = await self._execute(frame, snapshot, decision)
                        except StaleActionError as exc:
                            # 换牌倒计时已结束时只能跳过旧画面，随后仍需进入正式
                            # 游戏循环，不能让一次错过换牌导致整局自动游戏退出。
                            LOGGER.warning("八雀启动换牌画面已过期：%s", exc)
                        else:
                            self.action_count += 1
                            self._last_action_fingerprint = snapshot.fingerprint()
                            self._last_action_at = time.monotonic()
                            LOGGER.info("八雀启动流程：%s", message)
                return snapshot

            # 开始按钮点击后会经历发牌/动画/等待阶段；这期间不误点。
            await asyncio.sleep(0.8)

        raise TimeoutError(
            "点击“开始游戏”后 90 秒内仍未识别到正式牌局，已停止自动启动。"
        )

    async def run_loop(self, stop_event: Event | None = None) -> str:
        unknown_streak = 0
        stop_signal = stop_event or self.session.stop_event
        try:
            while not stop_signal.is_set():
                # 有独立观察器（或本地视觉暂停对手识别）时，主循环无需逐帧
                # 比较对手动作，可只在自己回合调用视觉模型。
                turn_gated = self.external_observer_active
                try:
                    prefetched = (
                        await self._wait_for_turn(stop_signal) if turn_gated else None
                    )
                    if stop_signal.is_set():
                        break
                    result = await self.step(execute=True, prefetched=prefetched)
                except RuntimeError as exc:
                    text = str(exc)
                    if any(
                        marker in text
                        for marker in (
                            "前台窗口",
                            "没有可用的前台窗口",
                            "没有找到包含",
                        )
                    ):
                        LOGGER.warning("游戏窗口暂不可用，本轮等待后重试：%s", exc)
                        await asyncio.sleep(self.action_interval_seconds)
                        continue
                    raise
                self.last_message = result.message
                LOGGER.info(
                    "八雀自动游戏 acted=%s state=%s message=%s",
                    result.acted,
                    result.snapshot.screen_type.value,
                    result.message,
                )
                if result.stop_loop:
                    return result.message
                if result.snapshot.screen_type == ScreenType.UNKNOWN:
                    unknown_streak += 1
                else:
                    unknown_streak = 0
                if unknown_streak >= 6:
                    return "连续 6 次无法可靠识别游戏状态，为避免误点已停止。"
                # 回合内识别被拒（漏牌等）时立即重探，别把倒计时浪费在固定间隔上。
                await asyncio.sleep(
                    self.action_interval_seconds
                    if result.acted or not turn_gated
                    else 0.2
                )
            return "已收到停止信号。"
        except asyncio.CancelledError:
            return "自动游戏已停止。"
        except Exception as exc:
            LOGGER.exception("八雀自动游戏异常")
            return f"自动游戏停止：{type(exc).__name__}: {exc}"

    def state_text(self) -> str:
        if self.last_snapshot is None:
            return "尚未读取游戏画面。"
        return f"{self._snapshot_summary(self.last_snapshot)}\n最近动作：{self.last_message}\n累计动作：{self.action_count}"

    def quality_metrics(self) -> dict[str, object]:
        unknown_rate = (
            self.unknown_snapshot_count / self.inspect_count
            if self.inspect_count
            else None
        )
        rejection_rate = (
            self.validation_rejection_count / self.step_count
            if self.step_count
            else None
        )
        return {
            "inspect_count": self.inspect_count,
            "unknown_snapshot_count": self.unknown_snapshot_count,
            "unknown_rate": round(unknown_rate, 4) if unknown_rate is not None else None,
            "step_count": self.step_count,
            "validation_rejection_count": self.validation_rejection_count,
            "validation_rejection_rate": (
                round(rejection_rate, 4) if rejection_rate is not None else None
            ),
            "action_count": self.action_count,
            "opponent_event_count": self.opponent_event_count,
            "tracker_started": self._tracker_started,
            "external_observer_active": self.external_observer_active,
            "fast_ui_count": self.fast_ui_count,
            "qwen_vision_count": self.qwen_vision_count,
            "fast_ui_fallback_count": self.fast_ui_fallback_count,
            "fast_vision_enabled": self.fast_vision_enabled,
            "decision_mode": self.decision_mode,
            "hybrid_decision_count": self.hybrid_decision_count,
            "hybrid_fallback_count": self.hybrid_fallback_count,
            "last_decision_source": self.last_decision_source,
        }

    def _decide(
        self,
        snapshot: GameSnapshot,
        tracker: TableTracker | None = None,
    ) -> GameDecision:
        tracker = tracker or self.tracker

        if snapshot.screen_type == ScreenType.SWAP:
            self.last_decision_source = "rule"
            return choose_swap_two(snapshot.hand, snapshot.gold_ranks)

        if self.decision_mode == "rule":
            self.last_decision_source = "rule"
            if snapshot.screen_type == ScreenType.DRAW:
                return choose_draw(
                    snapshot.hand,
                    snapshot.visible_cards,
                    snapshot.gold_ranks,
                )
            if snapshot.screen_type in {ScreenType.DISCARD, ScreenType.HU}:
                return choose_discard(snapshot.hand, snapshot.gold_ranks)
            raise RuntimeError(f"当前状态不可决策：{snapshot.screen_type.value}")

        if snapshot.screen_type == ScreenType.DRAW:
            # 摸牌阶段候选空间较小，继续由 Python EV 决定，避免多一次模型延迟。
            self.last_decision_source = "ev"
            return self.ev.choose_draw_action(
                snapshot.hand,
                snapshot.visible_cards,
                snapshot.gold_ranks,
                tracker,
                opportunity_available=snapshot.opportunity_click is not None,
                time_budget_seconds=self.decision_budget_seconds,
            )

        if snapshot.screen_type in {ScreenType.DISCARD, ScreenType.HU}:
            decision, values = self.ev.choose_8_card_action(
                snapshot.hand,
                snapshot.gold_ranks,
                tracker,
                time_budget_seconds=self.decision_budget_seconds,
            )
            LOGGER.info(
                "八雀EV候选 %s",
                [
                    {
                        "action": item.action,
                        "card_index": item.card_index,
                        "card": (
                            snapshot.hand[item.card_index].label
                            if item.card_index is not None
                            and 0 <= item.card_index < len(snapshot.hand)
                            else None
                        ),
                        "ev": round(item.expected_value, 2),
                        "future": round(item.future_hu_value, 2),
                        "risk": round(item.opponent_risk, 2),
                    }
                    for item in values[:5]
                ],
            )

            if self.decision_mode == "hybrid" and self.decision_reranker is not None:
                hybrid = self.decision_reranker.rerank_8_card_action(
                    hand=snapshot.hand,
                    gold_ranks=snapshot.gold_ranks,
                    tracker=tracker,
                    ev_decision=decision,
                    values=values,
                )
                if hybrid.used_model:
                    self.hybrid_decision_count += 1
                    self.last_decision_source = "hybrid_qwen"
                    LOGGER.info(
                        "八雀HYBRID决策 choice=%s confidence=%.2f action=%s indexes=%s reason=%s",
                        hybrid.choice_id,
                        hybrid.confidence,
                        hybrid.decision.action,
                        hybrid.decision.card_indexes,
                        hybrid.reason,
                    )
                    decision = hybrid.decision
                else:
                    self.hybrid_fallback_count += 1
                    self.last_decision_source = "hybrid_ev_fallback"
                    LOGGER.info("八雀HYBRID回退EV：%s", hybrid.reason)
            else:
                self.last_decision_source = "ev"
                if self.decision_mode == "hybrid":
                    self.hybrid_fallback_count += 1
                    self.last_decision_source = "hybrid_ev_fallback"
                    LOGGER.info("八雀HYBRID未配置本地决策模型，沿用Python EV")

            if snapshot.screen_type == ScreenType.HU and decision.action != "hu":
                LOGGER.info("当前虽可胡，但最终策略选择继续做牌：%s", decision.reason)
            return decision

        raise RuntimeError(f"当前状态不可决策：{snapshot.screen_type.value}")

    async def _execute(
        self,
        frame: CapturedFrame,
        snapshot: GameSnapshot,
        decision: GameDecision,
    ) -> str:
        if decision.action == "swap":
            await self._require_action_state("swap")
            assert snapshot.swap_button is not None
            desired = set(decision.card_indexes)
            for index, card in enumerate(snapshot.hand):
                if card.selected and index not in desired:
                    LOGGER.info(
                        "八雀换牌：取消误选 index=%s card=%s point=(%s,%s)",
                        index,
                        card.label,
                        card.x,
                        card.y,
                    )
                    await self._click_card(frame, card)
                    await asyncio.sleep(0.30)
            for index in decision.card_indexes:
                card = snapshot.hand[index]
                if not card.selected:
                    LOGGER.info(
                        "八雀换牌：选择 index=%s card=%s point=(%s,%s)",
                        index,
                        card.label,
                        card.x,
                        card.y,
                    )
                    await self._click_card(frame, card)
                    # 选中的牌会先抬高；等待动画结束再选择下一张，避免游戏吞掉
                    # 过快的连续点击或把同一张牌误切回未选状态。
                    await asyncio.sleep(0.30)
            LOGGER.info(
                "八雀换牌：两张牌已选择，点击换牌按钮 point=(%s,%s)",
                snapshot.swap_button.x,
                snapshot.swap_button.y,
            )
            # 两张牌选中后灰色“换牌”按钮会变为橙色，本地颜色门会把它归为
            # discard。此处允许 swap/discard 两种“仍有蓝色操作横幅”的状态；
            # None（对手回合或倒计时结束）仍会立即取消。
            await self._require_any_action_state({"swap", "discard"}, "换牌确认")
            await self._click(frame, snapshot.swap_button)
            return decision.reason + "；已点击换牌。"

        if decision.action == "draw_visible":
            await self._require_action_state("draw")
            if decision.visible_index is None:
                raise RuntimeError("draw_visible 缺少 visible_index")
            card = snapshot.visible_cards[decision.visible_index]
            await self._click_card(frame, card)
            self.tracker.record_own_take_visible(card)
            return decision.reason + "；已取亮牌。"

        if decision.action == "draw_opportunity":
            await self._require_action_state("draw")
            if snapshot.opportunity_click is None:
                raise RuntimeError("决策使用问号选牌，但未识别到手中问号牌坐标")
            if decision.visible_index is not None:
                if not 0 <= decision.visible_index < len(snapshot.visible_cards):
                    raise RuntimeError("问号选牌的亮牌索引超出范围")
                target = snapshot.visible_cards[decision.visible_index]
                await self._click_card(frame, target)
            else:
                if snapshot.deck_click is None:
                    raise RuntimeError("问号选牌选择牌堆，但未识别到牌堆坐标")
                await self._click(frame, snapshot.deck_click)
            # 问号牌是对中央目标的确认，不是第一步入口。等选中动画结束后
            # 再确认仍有操作权，避免用旧截图坐标误点下一回合。
            await asyncio.sleep(0.30)
            await self._require_action_state("draw")
            await self._click(frame, snapshot.opportunity_click)
            if decision.visible_index is not None:
                self.tracker.record_own_take_visible(
                    snapshot.visible_cards[decision.visible_index]
                )
            else:
                self.tracker.record_own_draw_deck()
            return decision.reason + "；已按候选牌→问号牌顺序完成选牌。"

        if decision.action == "draw_deck":
            await self._require_action_state("draw")
            if snapshot.deck_click is None:
                raise RuntimeError("未识别到牌堆点击位置")
            await self._click(frame, snapshot.deck_click)
            self.tracker.record_own_draw_deck()
            return decision.reason + "；已摸牌堆。"

        if decision.action == "hu":
            await self._require_action_state("discard")
            if snapshot.hu_button is None:
                raise RuntimeError(
                    f"规则引擎判断已胡（{decision.hu.pattern if decision.hu else '未知'}），"
                    "但视觉没有可靠识别到胡牌按钮；为避免把胡牌打掉，本轮不操作。"
                )
            await self._click(frame, snapshot.hu_button)
            return decision.reason + "；已点击胡牌。"

        if decision.action == "discard":
            await self._require_action_state("discard")
            if snapshot.discard_button is None:
                raise RuntimeError("未识别到出牌按钮")
            index = decision.card_indexes[0]
            desired = snapshot.hand[index]
            for i, card in enumerate(snapshot.hand):
                if card.selected and i != index:
                    await self._click_card(frame, card)
                    await asyncio.sleep(0.1)
            if not desired.selected:
                await self._click_card(frame, desired)
                await asyncio.sleep(0.30)
            # 选牌后再次读取即时画面。若倒计时已过或轮到对手，绝不能继续
            # 用旧截图里的“出牌”按钮坐标点击。
            await self._require_action_state("discard")
            await self._click(frame, snapshot.discard_button)
            self.tracker.record_own_discard(desired)
            return decision.reason + "；已出牌。"

        raise RuntimeError(f"未知游戏动作：{decision.action}")

    async def _current_action_state(self) -> str | None:
        for attempt in range(2):
            try:
                return await asyncio.to_thread(
                    current_game_action_state,
                    self.window_title_keyword,
                )
            except RuntimeError as exc:
                LOGGER.warning(
                    "读取即时操作权失败，尝试重新聚焦 attempt=%s error=%s",
                    attempt + 1,
                    exc,
                )
                await asyncio.to_thread(focus_game_window, self.window_title_keyword)
                await asyncio.sleep(0.15)
        return None

    async def _require_action_state(self, expected: str) -> None:
        actual = await self._current_action_state()
        if actual != expected:
            raise StaleActionError(
                f"操作前即时复核失败：期望 {expected}，当前为 "
                f"{actual or '对手回合/无操作权'}，已取消点击。"
            )

    async def _require_any_action_state(
        self,
        expected: set[str],
        label: str,
    ) -> None:
        actual = await self._current_action_state()
        if actual not in expected:
            raise StaleActionError(
                f"{label}前即时复核失败：当前为 "
                f"{actual or '对手回合/无操作权'}，已取消点击。"
            )

    def opponent_state_text(self) -> str:
        if not self._tracker_started:
            return "对手记牌：尚未建立本局状态。"
        lines = [f"对手记牌：已接收 {self.opponent_event_count} 个实时事件"]
        for seat, data in self.tracker.opponent_known_summary().items():
            known = data["known_hand"]
            known_text = "、".join(f"{key}×{count}" for key, count in known.items()) or "无"
            lines.append(
                f"- {seat}: 已知存牌[{known_text}]，未知{data['unknown_hand_count']}张，"
                f"已记录出牌{len(data['discards'])}张"
            )
        if self._round_gold_ranks:
            inferences = self.ev.infer_opponents(
                self.tracker,
                self._round_gold_ranks,
                samples=max(120, self.ev.simulations // 3),
                top_patterns=4,
                top_cards=6,
            )
            lines.append("对手可能牌型：")
            for item in inferences:
                if item.patterns:
                    pattern_text = "；".join(
                        f"{p.pattern} {p.probability_next_draw:.1%}"
                        for p in item.patterns[:4]
                    )
                else:
                    pattern_text = "暂未形成明确的一摸成牌方向"
                danger_text = "、".join(
                    f"{key}({prob:.1%})"
                    for key, prob in item.dangerous_cards[:5]
                ) or "无"
                lines.append(
                    f"- {item.seat}: 下一摸任意胡 {item.any_hu_probability_next_draw:.1%}；"
                    f"{pattern_text}；危险牌 {danger_text}"
                )
        return "\n".join(lines)

    def _apply_opponent_events(self, snapshot: GameSnapshot) -> None:
        if not self._tracker_started or not snapshot.opponent_events:
            return

        now = time.monotonic()
        ttl_seconds = 2.5
        self._pair_event_dedupe = {
            signature: seen_at
            for signature, seen_at in self._pair_event_dedupe.items()
            if now - seen_at <= ttl_seconds
        }

        for event in snapshot.opponent_events:
            if event.confidence < self.min_event_confidence:
                LOGGER.info(
                    "忽略低置信度对手事件 seat=%s action=%s confidence=%.2f",
                    event.seat,
                    event.action,
                    event.confidence,
                )
                continue

            card_key = "NONE"
            if event.card is not None:
                card_key = "WILD" if event.card.wild else f"{event.card.rank}:{event.card.suit}"
            signature = (event.seat, event.action, card_key)
            if signature in self._pair_event_dedupe:
                continue

            try:
                if event.action == "draw_deck":
                    self.tracker.record_opponent_draw_deck(event.seat)
                elif event.action == "take_visible":
                    if event.card is None:
                        LOGGER.warning("拿亮牌事件缺少牌，已忽略 seat=%s", event.seat)
                        continue
                    self.tracker.record_opponent_take_visible(event.seat, event.card)
                elif event.action == "discard":
                    if event.card is None:
                        LOGGER.warning("出牌事件缺少牌，已忽略 seat=%s", event.seat)
                        continue
                    self.tracker.record_opponent_discard(event.seat, event.card)
                elif event.action == "finished":
                    self.tracker.mark_finished(event.seat)
                else:
                    LOGGER.warning("未知对手事件 action=%s", event.action)
                    continue
            except (ValueError, RuntimeError) as exc:
                LOGGER.warning(
                    "对手事件与当前记牌状态冲突，已忽略 seat=%s action=%s card=%s error=%s",
                    event.seat,
                    event.action,
                    card_key,
                    exc,
                )
                continue

            self._pair_event_dedupe[signature] = now
            self.opponent_event_count += 1
            LOGGER.info(
                "写入对手事件 seat=%s action=%s card=%s confidence=%.2f detail=%s",
                event.seat,
                event.action,
                card_key,
                event.confidence,
                event.detail,
            )

    def _sync_tracker(self, snapshot: GameSnapshot) -> None:
        if len(snapshot.gold_ranks) != 3 or len(snapshot.hand) not in {7, 8}:
            return
        new_gold = tuple(snapshot.gold_ranks)
        should_reset = (
            not self._tracker_started
            or (self._round_gold_ranks and new_gold != self._round_gold_ranks)
            or (
                snapshot.deck_count is not None
                and self.tracker.deck_count is not None
                and snapshot.deck_count >= 55
                and self.tracker.deck_count < 20
            )
        )
        if should_reset:
            self.tracker.reset_round(snapshot.hand, snapshot.deck_count)
            self._tracker_started = True
            self._round_gold_ranks = new_gold
            self._pair_event_dedupe.clear()
            self._observation_dedupe.clear()
            self._observer_recent_events.clear()
            self.opponent_event_count = 0
            LOGGER.info("建立新一局记牌状态 gold=%s deck=%s", new_gold, snapshot.deck_count)
            return
        self.tracker.update_own_hand(snapshot.hand)
        self.tracker.set_deck_count(snapshot.deck_count)

    def apply_opponent_observation(self, observation: TableObservation) -> bool:
        """把高置信度的三家动作写入 TableTracker。

        观察器只负责“看到什么”；这里负责保守落账。任何关键牌看不清时宁可不写，
        避免一次误识别污染整局的剩余牌池。
        """
        if not self._tracker_started:
            LOGGER.info("记牌器尚未初始化，暂不采用对手事件：%s", observation.summary())
            return False
        if observation.actor not in {"left", "top", "right"}:
            return False

        signature = (
            observation.actor,
            observation.event_type,
            observation.draw_source,
            observation.drawn_card.rank,
            observation.drawn_card.suit,
            observation.discard_card.rank,
            observation.discard_card.suit,
            observation.opportunity_chosen_index,
            observation.deck_after,
        )
        now = time.monotonic()
        while self._observation_dedupe and now - self._observation_dedupe[0][0] > 5.0:
            self._observation_dedupe.popleft()
        if any(old_signature == signature for _, old_signature in self._observation_dedupe):
            LOGGER.info("跳过重复对手事件：%s", observation.summary())
            return False

        applied = False
        if observation.event_type == "turn":
            discard = observation.discard_card.to_card()
            if discard is None or observation.discard_card.confidence < 0.76:
                return False
            drawn = observation.drawn_card.to_card()
            draw_source = observation.draw_source
            if draw_source == "visible" and (
                drawn is None or observation.drawn_card.confidence < 0.72
            ):
                # 明确知道“拿了亮牌”但看不清具体牌时，不伪造成某张已知牌。
                draw_source = "unknown"
                drawn = None
            self.tracker.record_opponent_turn(
                observation.actor,
                draw_source=draw_source,
                drawn_card=drawn,
                discard=discard,
            )
            applied = True

        elif observation.event_type == "opportunity":
            options = [card.to_card() for card in observation.opportunity_cards]
            if (
                len(options) == 4
                and all(card is not None for card in options)
                and 0 <= observation.opportunity_chosen_index < 4
            ):
                self.tracker.record_opportunity_reveal(
                    observation.actor,
                    [card for card in options if card is not None],
                    observation.opportunity_chosen_index,
                )
                applied = True

        elif observation.event_type == "hu":
            self.tracker.mark_finished(observation.actor, True)
            applied = True

        if not applied:
            return False

        if observation.deck_after >= 0:
            self.tracker.set_deck_count(observation.deck_after)
        self._observation_dedupe.append((now, signature))
        self._observer_recent_events.append(observation.summary())
        self.opponent_event_count += 1
        LOGGER.info("已采用对手事件：%s", observation.summary())
        return True

    def set_external_observer_active(self, active: bool) -> None:
        self.external_observer_active = bool(active)
        if active:
            self._previous_frame_path = None

    def recent_opponent_events_text(self) -> str:
        if not self._observer_recent_events:
            return "最近对手事件：暂无"
        return "最近对手事件：\n" + "\n".join(
            f"- {item}" for item in list(self._observer_recent_events)[-6:]
        )

    async def _click_card(self, frame: CapturedFrame, card: Card) -> None:
        if card.x is None or card.y is None:
            raise RuntimeError("牌缺少点击坐标")
        await self._click(frame, Point(card.x, card.y))

    async def _click(self, frame: CapturedFrame, point: Point) -> None:
        if not (0 <= point.x < frame.width and 0 <= point.y < frame.height):
            raise RuntimeError(
                f"视觉坐标超出游戏截图：({point.x},{point.y}) / {frame.width}x{frame.height}"
            )
        # Windows 在窗口切换动画期间偶尔会短暂返回空前台句柄。自动模式应尝试
        # 重新聚焦游戏，而不是因为一次瞬时 GetForegroundWindow()==0 终止整局。
        foreground_error: Exception | None = None
        for attempt in range(3):
            try:
                await asyncio.to_thread(assert_game_foreground, self.window_title_keyword)
                foreground_error = None
                break
            except RuntimeError as exc:
                foreground_error = exc
                LOGGER.warning(
                    "点击前游戏窗口未就绪，尝试重新聚焦 attempt=%s error=%s",
                    attempt + 1,
                    exc,
                )
                await asyncio.to_thread(focus_game_window, self.window_title_keyword)
                await asyncio.sleep(0.20)
        if foreground_error is not None:
            raise RuntimeError("无法在点击前恢复游戏前台窗口") from foreground_error
        screen_x = frame.origin_x + point.x
        screen_y = frame.origin_y + point.y

        self.session.set_native_call(True)
        try:
            await asyncio.to_thread(click_screen_native, screen_x, screen_y)
            LOGGER.info(
                "八雀原生点击 relative=(%s,%s) absolute=(%s,%s)",
                point.x,
                point.y,
                screen_x,
                screen_y,
            )
        finally:
            self.session.set_native_call(False)

    def _validate_actionable_snapshot(
        self,
        snapshot: GameSnapshot,
        frame: CapturedFrame,
    ) -> str | None:
        if len(snapshot.gold_ranks) != 3:
            return f"金牌点数应为 3 个，当前只可靠识别到 {len(snapshot.gold_ranks)} 个，暂不操作。"

        expected = {
            ScreenType.DRAW: 6 if snapshot.opportunity_click is not None else 7,
            ScreenType.DISCARD: 8,
            ScreenType.HU: 8,
        }.get(snapshot.screen_type)
        if snapshot.screen_type == ScreenType.SWAP and len(snapshot.hand) not in {6, 7}:
            return (
                "swap 阶段应识别到 6 或 7 张可操作手牌，"
                f"当前为 {len(snapshot.hand)} 张，暂不操作。"
            )
        if expected is not None and len(snapshot.hand) != expected:
            return (
                f"{snapshot.screen_type.value} 阶段应识别到 {expected} 张手牌，"
                f"当前为 {len(snapshot.hand)} 张，暂不操作。"
            )

        for card in [*snapshot.hand, *snapshot.visible_cards]:
            if card.confidence < self.min_card_confidence:
                return f"存在低置信度牌 {card.label}（{card.confidence:.2f}），暂不操作。"
            if card.x is None or card.y is None:
                return "存在缺少坐标的牌，暂不操作。"
            if not (0 <= card.x < frame.width and 0 <= card.y < frame.height):
                return f"牌坐标超出截图范围：{card.label} ({card.x},{card.y})"

        if snapshot.screen_type == ScreenType.SWAP and snapshot.swap_button is None:
            return "换牌阶段未识别到“换牌”按钮，暂不操作。"
        if (
            snapshot.screen_type == ScreenType.DRAW
            and snapshot.opportunity_click is None
            and snapshot.deck_click is None
            and not snapshot.visible_cards
        ):
            return "摸牌阶段未识别到问号选牌、牌堆或普通亮牌，暂不操作。"
        if snapshot.screen_type == ScreenType.DISCARD and snapshot.discard_button is None:
            return "出牌阶段未识别到“出牌”按钮，暂不操作。"
        return None

    @staticmethod
    def _snapshot_summary(snapshot: GameSnapshot) -> str:
        hand = " ".join(card.label + ("*" if card.selected else "") for card in snapshot.hand) or "无"
        visible = " ".join(card.label for card in snapshot.visible_cards) or "无"
        opportunity = (
            "有"
            if snapshot.opportunity_click is not None
            else "无"
        )
        gold = "/".join(snapshot.gold_ranks) or "未知"
        deck = "未知" if snapshot.deck_count is None else str(snapshot.deck_count)
        return (
            f"状态：{snapshot.screen_type.value}\n"
            f"金牌：{gold}\n"
            f"手牌({len(snapshot.hand)})：{hand}\n"
            f"亮牌：{visible}\n"
            f"问号选牌：{opportunity}\n"
            f"牌堆：{deck}"
        )

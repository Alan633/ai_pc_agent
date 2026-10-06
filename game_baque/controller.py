from __future__ import annotations

import asyncio
import json
from datetime import datetime
import copy
import logging
import math
import time
from collections import Counter, deque
from collections.abc import Awaitable, Callable
from threading import Event
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image

from agent.session import AgentSession
from game_baque.capture import (
    CapturedFrame,
    assert_game_foreground,
    capture_game_window,
    capture_game_window_if_swap,
    click_screen_native,
    capture_game_window,
    current_action_buttons,
    current_game_action_state,
    current_start_page_check,
    is_start_page_button,
    focus_game_window,
    probe_game_turn,
    table_signature_changed,
    TurnProbe,
)
from game_baque.ev import ExpectedValueEngine
from game_baque.event_vision import TableObservation
from game_baque.fast_vision import analyze_image_path, detect_table_hu_badges
from game_baque.local_decision import LocalDecisionReranker
from game_baque.models import Card, GameDecision, GameSnapshot, Point, ScreenType
from game_baque.plan_report import opponent_plan_lines, own_plan_line, target_line
from game_baque.suit_glyph import classify_hand_suits
from game_baque.table_watch import OpponentTurn, RowDiffTracker
from game_baque.round_report import RoundLog, RoundReport
from game_baque.rules import evaluate_hu
from game_baque.strategy import choose_discard, choose_draw, choose_swap_two
from game_baque.tracker import TableTracker, card_from_key, card_key
from game_baque.vision import (
    TURN_DISCARD_BUTTON,
    TURN_HAND_SLOTS,
    TURN_HAND_Y,
    BaqueVision,
)

LOGGER = logging.getLogger(__name__)

# 一个回合（摸牌+出牌）共用约 20 秒倒计时；点击与操作前复核约需 2 秒。
TURN_SECONDS = 20.0
ACTION_MARGIN_SECONDS = 2.0
# 牌局中途金牌读数变化：需连续两次读到同一组新金牌，或牌堆仍接近满堆（真新局）才接受。
GOLD_CHANGE_CONFIRMATIONS = 2
FRESH_ROUND_DECK_COUNT = 50
# 选牌弹窗里估算“未选中三张回到牌堆顶被对手摸到”风险的计算上限（秒）。
CHOICE_RISK_SECONDS = 1.0
# 一个普通平胡（3+3+2、无金牌、无癞子付两次）约 220 理论分：三家每人能付的都不超过
# 这个量时，平胡已能收满，改为平胡优先。
PINGHU_SATURATION = 300.0
# 本地模型重排还要等它最坏超时；回合至少再富余这么多秒才调用，否则直接用 EV。
HYBRID_SPARE_SECONDS = 4.0
# 操作前复核失败时的重试窗口：对手扔的互动表情会短暂遮住操作横幅。
STATE_RECHECK_SECONDS = 1.2
# 出牌后播报牌型计划的计算上限（秒，自己/三家各一份）；在对手回合进行，不占自己回合时间。
PLAN_LOG_SECONDS = 0.6
# 像素门“换牌”需持续这么久才算新一局；摸牌飞牌动画会短暂盖住按钮区被读成灰色换牌钮。
SWAP_CONFIRM_SECONDS = 0.8


@dataclass(frozen=True, slots=True)
class GameStepResult:
    acted: bool
    message: str
    snapshot: GameSnapshot
    frame: CapturedFrame
    stop_loop: bool = False


class StaleActionError(RuntimeError):
    """模型返回后画面已换回合，当前动作必须丢弃。"""


_FOREGROUND_ERROR_MARKERS = ("前台窗口", "没有可用的前台窗口", "没有找到包含")


def _is_foreground_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(marker in text for marker in _FOREGROUND_ERROR_MARKERS)


class BaqueGameController:
    def __init__(
        self,
        *,
        vision: BaqueVision,
        screenshots_dir: Path,
        session: AgentSession,
        window_title_keyword: str = "欢乐斗地主",
        action_interval_seconds: float = 0.6,
        min_card_confidence: float = 0.72,
        min_event_confidence: float = 0.80,
        auto_next_round: bool = False,
        decision_budget_seconds: float = 1.2,
        fast_vision_enabled: bool = True,
        decision_mode: str = "ev",
        decision_reranker: LocalDecisionReranker | None = None,
        win_cap_beans: float = 0.0,
        rounds_log_path: Path | None = None,
    ) -> None:
        self.vision = vision
        # 截图根目录；每局开始后 screenshots_dir 切到 根目录/rounds/<开局时间>/，
        # 该局所有截图（开始页、换牌、各回合、弹窗、结算）单独存放。
        self.base_screenshots_dir = screenshots_dir
        self.screenshots_dir = screenshots_dir
        self.session = session
        self.window_title_keyword = window_title_keyword
        self.action_interval_seconds = max(0.2, float(action_interval_seconds))
        self.min_card_confidence = min(max(float(min_card_confidence), 0.0), 1.0)
        self.min_event_confidence = min(max(float(min_event_confidence), 0.0), 1.0)
        self.auto_next_round = bool(auto_next_round)
        self.fast_vision_enabled = bool(fast_vision_enabled)
        normalized_mode = decision_mode.strip().lower()
        if normalized_mode not in {"rule", "ev", "hybrid"}:
            raise ValueError("decision_mode 必须是 rule、ev 或 hybrid")
        self.decision_mode = normalized_mode
        self.decision_reranker = decision_reranker
        # 回合约 20 秒；实时 EV 默认用 2 秒，在稳定性与操作响应之间取平衡。
        self.decision_budget_seconds = max(0.5, float(decision_budget_seconds))
        # 单局输赢封顶手动覆盖（豆，0=按场次自动）；配合底分、对手豆数换算胡牌价值上限。
        self.win_cap_beans = max(0.0, float(win_cap_beans))
        self._round_info_attempts = 0
        # 每局结果记录：衡量胜率、定位丢失回合的原因。
        self.round_log = RoundLog(rounds_log_path)
        self.round_report: RoundReport | None = None
        self._turn_discarded = False
        self._turn_problem = ""
        self._turn_counted = False
        # 本回合刚从桌面拿进的亮牌：出牌时不再打回。
        self._taken_visible: Card | None = None
        # 本回合首次探测到操作权的时刻；对手回合时清空。
        self._turn_started_at: float | None = None
        # 本回合尝试打出“?”选牌卡的次数，防止点击未生效时反复抬起/放下。
        self._opportunity_attempts = 0
        # 开局换牌阶段：选中两张后“换牌”按钮变橙，像素门会读成 discard，且最左侧
        # 常是“?”选牌卡；此阶段绝不能去“打出选牌卡”。见到对手回合或摸牌后结束。
        self._swap_phase_active = True
        # 最近一次可靠读到的三张金牌；回合内快速识别直接复用，不再让模型重读。
        self._last_gold_ranks: tuple[str, ...] = ()
        self.last_snapshot: GameSnapshot | None = None
        self.last_message = "尚未识别"
        self.action_count = 0
        self.tracker = TableTracker()
        self.ev = ExpectedValueEngine()
        self._tracker_started = False
        self._round_gold_ranks: tuple[str, ...] = ()
        # 尚未确认的金牌新读数及连续命中次数；单帧误读（如 8→6）不能清空记牌。
        self._pending_gold_ranks: tuple[str, ...] = ()
        self._pending_gold_hits = 0
        # 牌局中途像素门首次报告“换牌”的时刻，用于过滤动画造成的瞬时误判。
        self._swap_probe_since: float | None = None
        # 对手回合记牌：比较亮牌行前后变化（见 table_watch）。
        self._row_diff = RowDiffTracker()
        # 本回合摸牌阶段看到的亮牌行/牌堆数与自己的取牌方式，出牌后据此推算对手行动前的状态。
        self._own_draw_row: tuple[list[Card], int | None, Card | None, bool] | None = None
        self._row_prev_signature: bytes | None = None
        self._row_read_signature: bytes | None = None
        # 胡牌结算横幅仍在显示：同一次结算只处理一次。
        self._settlement_active = False
        # 自己的欢乐豆（每局读本局信息时更新）：能赢多少也受它限制。
        self.own_balance: float | None = None
        # 已读到底分、确定同花顺优先/追大牌模式（换牌前需要）。
        self._mode_ready = False
        # 新一局换牌后还没给出“本局目标”。
        self._target_pending = False
        # 自己出牌后剩下的手牌（显示顺序，不含问号）与问号张数：对手回合里不会变，
        # 轮到自己摸牌时核对无误即可不再调用模型读手牌。
        self._known_hand: tuple[list[Card], int] | None = None
        # 对手回合最近一次读到的亮牌行（0-1000 坐标）与牌堆数。
        self._last_row_local: tuple[list[Card], int | None] | None = None
        # 与 _last_row_local 同序：每张亮牌上是否有游戏标的“胡”角标。
        self._last_row_hu: tuple[bool, ...] | None = None
        # 拿亮牌后预先算的出牌决策：(8 张牌键, 能否胡, 8 张原顺序, 后台任务)。
        self._discard_precompute: tuple[object, ...] | None = None
        # 对手回合里预先算好的摸牌决策：(局面键, 后台任务)。
        self._draw_precompute: tuple[tuple[object, ...], asyncio.Future[GameDecision]] | None = None
        self._last_action_fingerprint: tuple[object, ...] | None = None
        self._last_action_at = 0.0
        self._observation_dedupe: deque[tuple[float, tuple[object, ...]]] = deque(maxlen=24)
        self._observer_recent_events: deque[str] = deque(maxlen=12)
        self._previous_frame_path: Path | None = None
        self._pair_event_dedupe: dict[tuple[str, str, str], float] = {}
        self.external_observer_active = False
        self._pause_opponent_observer: Callable[[], Awaitable[None]] | None = None
        self._resume_opponent_observer: Callable[[], Awaitable[None]] | None = None
        self._observer_phase = "inactive"
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

    @staticmethod
    def _hu_offered(snapshot: GameSnapshot) -> bool:
        """游戏只在“7 张手牌再摸进一张凑成牌型”时给出“胡”按钮。

        8 张满足牌型不等于能胡：例如用“?”选牌补成的 8 张就没有胡按钮，
        必须先打出一张、等下一次摸牌。因此以游戏实际显示的胡按钮为准。
        """
        return snapshot.screen_type == ScreenType.HU or snapshot.hu_button is not None

    @staticmethod
    def _hand_slot_count(snapshot: GameSnapshot) -> int:
        count = len(snapshot.hand) + snapshot.opportunity_card_count
        # 兼容旧视觉格式：换牌页只返回 6 张真实牌，但没有显式返回最左侧问号牌。
        if (
            snapshot.screen_type == ScreenType.SWAP
            and len(snapshot.hand) == 6
            and snapshot.opportunity_card_count == 0
        ):
            return 7
        return count

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

        if fast is not None and fast.screen_type in {ScreenType.LOBBY, ScreenType.RECHARGE}:
            self.fast_ui_count += 1
            snapshot = GameSnapshot(
                screen_type=fast.screen_type,
                message=f"{fast.message} confidence={fast.confidence:.2f}",
            )
            LOGGER.info(
                "八雀OpenCV命中 %s confidence=%.2f", fast.screen_type.value, fast.confidence
            )
            self._register_snapshot(frame, snapshot)
            return snapshot

        # 开始页完全不需要多模态模型：OpenCV 直接给出按钮中心。
        if (
            fast is not None
            and fast.screen_type == ScreenType.START
            and fast.start_button is not None
            and fast.confidence >= 0.72
            and await self._start_page_ok(frame, fast.start_button, live=False)
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
            snapshot = await self.vision.parse_turn(
                frame.path,
                fast.action_state,
                gold_ranks=self._last_gold_ranks,
            )
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
        if snapshot.screen_type == ScreenType.DRAW and snapshot.visible_hu is None:
            snapshot.visible_hu = self._row_hu_flags(frame.path, snapshot.visible_cards, local=False)
        if snapshot.screen_type == ScreenType.DRAW and self._row_diff.active:
            # 轮到自己摸牌时读到的亮牌行，补齐最后一家（上家）的动作。
            self._apply_row_read(snapshot.visible_cards, snapshot.deck_count)
        if snapshot.screen_type == ScreenType.DRAW and self._target_pending and snapshot.hand:
            self._announce_target(snapshot.hand)
        if snapshot.screen_type == ScreenType.DRAW:
            # 开局第一次轮到自己：把之前暂存的对手动作按座位倒推后写入记牌。
            for turn in self._row_diff.finish_opening():
                self._record_opponent_turn(turn)
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
            probed_at = time.monotonic()
            probe = await asyncio.to_thread(
                probe_game_turn,
                self.screenshots_dir,
                self.window_title_keyword,
            )
            if probe.settlement is not None:
                self._on_settlement(probe.settlement)
            else:
                self._settlement_active = False
            if probe.action_state is None:
                if probe.round_in_progress:
                    await self._set_observer_phase("opponent")
                    await self._watch_table(probe)
                    if self._target_pending:
                        await self._announce_target_from_idle()
                    if self._turn_counted:
                        self._end_turn()
                    self._turn_started_at = None
                    self._opportunity_attempts = 0
                    self._swap_phase_active = False
                    if self._round_info_attempts < 2:
                        # 对手回合空闲时读一次底分与对手豆数（约 1 秒），不占自己回合。
                        await self._set_observer_phase("own")
                        try:
                            await self._read_round_info()
                        finally:
                            await self._set_observer_phase("opponent")
                        continue
                else:
                    await self._set_observer_phase("inactive")
            else:
                # 先抢占/取消本地对手识别，再提交自己的牌面请求。
                await self._set_observer_phase("own")
                if self._turn_started_at is None:
                    self._turn_started_at = probed_at
            if probe.action_state == "swap" and not self._swap_phase_active:
                if self._swap_probe_since is None:
                    self._swap_probe_since = probed_at
                if probed_at - self._swap_probe_since < SWAP_CONFIRM_SECONDS:
                    # 飞牌动画只持续零点几秒；持续不足时不开新局、也不交给视觉按换牌识别。
                    await asyncio.sleep(0.2)
                    continue
                # 连续自动模式下见到稳定的换牌页即新一局开始。
                self._start_round_report("新一局换牌")
                self._swap_phase_active = True
            if probe.action_state != "swap":
                if self._swap_probe_since is not None:
                    LOGGER.info("八雀像素门：忽略瞬时“换牌”误判（持续不足 %.1fs）", SWAP_CONFIRM_SECONDS)
                self._swap_probe_since = None
            if probe.action_state == "draw":
                self._swap_phase_active = False
            if (
                probe.action_state in {"draw", "discard", "choose"}
                and not self._swap_phase_active
                and not self._turn_counted
            ):
                self._begin_turn()
            if (
                probe.frame is not None
                and probe.opportunity_card
                and not self._swap_phase_active
                and self._opportunity_attempts < 4
            ):
                # 出牌阶段手里有“?”选牌卡：先打出它换取四选一，再正常出牌。
                await self._play_opportunity_card(probe.frame, probe.opportunity_point)
                continue
            if probe.frame is not None and probe.action_state == "choose":
                await self._handle_choice_popup(probe.frame)
                continue
            if probe.frame is not None and probe.action_state == "draw":
                fast = await self._fast_draw_snapshot(probe.frame)
                if fast is not None:
                    self._register_snapshot(probe.frame, fast)
                    return probe.frame, fast
            if probe.frame is not None and probe.action_state is not None:
                started_at = time.monotonic()
                try:
                    snapshot = await self.vision.parse_turn(
                        probe.frame.path,
                        probe.action_state,
                        gold_ranks=self._last_gold_ranks,
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

    async def _read_round_info(self) -> None:
        self._round_info_attempts += 1
        try:
            frame = await asyncio.to_thread(
                capture_game_window,
                self.screenshots_dir,
                self.window_title_keyword,
            )
            base, balances, room_cap = await self.vision.parse_round_info(frame.path)
        except (RuntimeError, ValueError, AttributeError) as exc:
            LOGGER.warning("八雀本局信息识别失败：%s", exc)
            return
        try:
            own = await self.vision.parse_own_beans(frame.path)
        except (RuntimeError, ValueError, AttributeError) as exc:
            LOGGER.warning("八雀自己豆数识别失败：%s", exc)
            own = None
        self._apply_payout_cap(base, balances, room_cap, own_balance=own)
        if self.round_report is not None:
            self.round_report.base_score = base
            self.round_report.opponent_beans = list(balances)
            self.round_report.room_cap = (
                self.win_cap_beans if self.win_cap_beans > 0 else room_cap
            )
        self._round_info_attempts = 2

    def _apply_payout_cap(
        self,
        base: float | None,
        balances: list[float],
        room_cap: float | None = None,
        *,
        own_balance: float | None = None,
    ) -> None:
        """每胡一次，每家最多支付 min(场次封顶, 该家欢乐豆, 自己欢乐豆)；换算为理论分（÷ 底分）。

        封顶按每胡一次、每一家分别计算（不是整局）：新手场每家每次最多 2 万；
        无癞子胡算两次支付，每家最多 2×封顶（仍不超过该家豆数）。
        以小博大有上限：自己只有 2.5 万豆时，胡再大的牌每家也最多付 2.5 万
        （两次支付合计也不超过）。反过来对手胡牌时，自己最多赔 min(封顶, 自己豆数)。
        """
        if own_balance is not None:
            self.own_balance = own_balance
        if base is not None:
            self._mode_ready = True
        own = self.own_balance
        per_seat_cap = self.win_cap_beans if self.win_cap_beans > 0 else room_cap
        seats: list[float | None] = list(balances[:3]) + [None] * (3 - min(3, len(balances)))
        seat_caps: list[float] = []
        seat_totals: list[float | None] = []
        for balance in seats:
            candidates = [value for value in (per_seat_cap, balance, own) if value is not None]
            # 某家豆数没读到时，只能按场次封顶/自己豆数估计；都未知则该家不封顶。
            seat_caps.append(min(candidates) if candidates else math.inf)
            totals = [value for value in (balance, own) if value is not None]
            seat_totals.append(min(totals) if totals else None)
        loss_candidates = [value for value in (per_seat_cap, own) if value is not None]
        self.ev.loss_cap = min(loss_candidates) / base if base and loss_candidates else None
        self.ev.loss_limit = own / base if base and own is not None else None
        # 同花顺优先：新手/普通场一律开启；高级/顶级场只有自己的豆够让对手付满封顶时
        # 才考虑特殊大牌型，否则也优先同花顺（听得宽、无癞子更好）。底分未知时按开启处理。
        self.ev.prefer_flush = not (
            base is not None
            and base >= 240
            and per_seat_cap is not None
            and own is not None
            and own >= per_seat_cap
        )
        # 豆够多（封顶不卡收入）时追大牌：全金牌、特殊牌型的结算倍数远高于普通同花顺。
        self.ev.chase_big = not self.ev.prefer_flush
        # 平胡优先：对手都没什么豆、一个普通平胡就能把他们能付的收满时，牌型再大也多收
        # 不到钱，胡得快、胡得多次才重要（优先于上面两种模式）。
        # 只看对手（用户：对手都没豆时）：每家对手这一胡最多能付的 min(封顶, 他的豆)，
        # 换算成理论分都不超过一个平胡。自己豆少不触发——那种情况按同花顺优先。
        opponent_payable = [
            min(value for value in (per_seat_cap, balance) if value is not None)
            for balance in seats
            if per_seat_cap is not None or balance is not None
        ]
        self.ev.prefer_pinghu = bool(
            base
            and len(opponent_payable) == 3
            and all(value / base <= PINGHU_SATURATION for value in opponent_payable)
        )
        if self.ev.prefer_pinghu:
            self.ev.prefer_flush = False
            self.ev.chase_big = False
        # 逐次模拟连胡要用的原始量：自己豆少时第一把被卡，但赢了之后上限会升高。
        self.ev.room_cap_theory = per_seat_cap / base if base and per_seat_cap else None
        self.ev.opponent_balances_theory = (
            tuple(balance / base for balance in balances[:3])
            if base and len(balances) >= 3 and all(b is not None for b in balances[:3])
            else None
        )
        self.ev.own_theory = own / base if base and own is not None else None
        # 连胡总额上限：三家对手的豆总数（不是“胡一次能收多少”）。
        known = [balance for balance in balances[:3] if balance is not None]
        self.ev.opponent_beans_total = (
            sum(known) / base if base and len(known) == 3 else None
        )
        if base is None or all(math.isinf(cap) for cap in seat_caps):
            self.ev.score_caps = None
            self.ev.seat_balances = None
            LOGGER.info("八雀本局封顶未知：底分=%s 对手豆=%s，按不封顶计算", base, balances)
            return
        self.ev.score_caps = tuple(max(1.0, cap / base) for cap in seat_caps)
        self.ev.seat_balances = tuple(
            total / base if total is not None else math.inf for total in seat_totals
        )
        finite = [cap for cap in seat_caps if not math.isinf(cap)]
        hard_caps = [
            min(2 * cap, total) if total is not None else (2 * cap if not math.isinf(cap) else None)
            for cap, total in zip(seat_caps, seat_totals)
        ]
        finite_hard = [cap for cap in hard_caps if cap is not None]
        LOGGER.info(
            "八雀封顶（每胡一次）：底分 %s，每家每次最多付 %s 豆（无癞子胡 ×2），对手豆 %s，"
            "自己豆 %s → 本次胡牌三家最多各付 %s 豆、合计 %s 豆；无癞子胡最多各付 %s 豆、合计 %s 豆；"
            "策略：%s",
            base,
            per_seat_cap or "未知",
            balances,
            own if own is not None else "未知",
            [round(cap) if not math.isinf(cap) else "不限" for cap in seat_caps],
            round(sum(finite)) if len(finite) == 3 else "不限",
            [round(cap) if cap is not None else "不限" for cap in hard_caps],
            round(sum(finite_hard)) if len(finite_hard) == 3 else "不限",
            "平胡优先（对手豆少，胡得快最重要）"
            if self.ev.prefer_pinghu
            else "同花顺优先"
            if self.ev.prefer_flush
            else "追大牌（豆够多，可考虑全金牌/特殊牌型）",
        )

    # ---- 每局结果记录 ----

    def _start_round_report(self, reason: str) -> None:
        if self.round_report is not None and self.round_report.turns > 0:
            self._finish_round_report(f"下一局开始（{reason}）")
        if self.round_report is None:
            self._new_round_report()
        self._turn_discarded = False
        self._turn_problem = ""

    def _new_round_report(self) -> RoundReport:
        """新建本局记录，并把截图目录切到本局专属文件夹。"""
        from datetime import datetime

        rounds_dir = self.base_screenshots_dir / "rounds"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        round_dir = rounds_dir / stamp
        suffix = 2
        while round_dir.exists():
            round_dir = rounds_dir / f"{stamp}-{suffix}"
            suffix += 1
        try:
            round_dir.mkdir(parents=True, exist_ok=False)
            self.screenshots_dir = round_dir
        except OSError as exc:
            LOGGER.warning("创建本局截图目录失败，继续使用 %s：%s", self.screenshots_dir, exc)
        self.round_report = RoundReport(screenshots_dir=str(self.screenshots_dir))
        LOGGER.info("八雀本局截图目录：%s", self.screenshots_dir)
        return self.round_report

    def _finish_round_report(self, reason: str) -> None:
        report = self.round_report
        if report is None:
            return
        if self._turn_counted:
            self._end_turn()
            self._turn_started_at = None
        if report.turns == 0 and not report.hu_clicked:
            # 还没轮到自己就结束（如开局前停止），不计入战绩。
            self.round_report = None
            return
        from datetime import datetime

        report.ended_at = datetime.now().isoformat(timespec="seconds")
        report.end_reason = reason
        report.gold_ranks = list(self._round_gold_ranks or self._last_gold_ranks)
        self.round_log.append(report)
        self.round_report = None

    def _begin_turn(self) -> None:
        if self.round_report is None:
            self._new_round_report()
        self.round_report.turns += 1
        self._turn_counted = True
        self._taken_visible = None
        self._turn_discarded = False
        self._turn_problem = ""

    def _end_turn(self) -> None:
        """回合结束：本回合没完成出牌/胡牌即记为丢失回合（多被系统代打）。"""
        report = self.round_report
        self._turn_counted = False
        self._taken_visible = None
        if report is None or report.turns == 0:
            return
        if not self._turn_discarded:
            report.lost_turns += 1
            report.add_reason(report.lost_turn_reasons, self._turn_problem or "未知（未完成出牌）")
        self._turn_discarded = False
        self._turn_problem = ""

    def _note_turn_problem(self, message: str, *, rejection: bool = False) -> None:
        self._turn_problem = message
        if rejection and self.round_report is not None:
            self.round_report.add_reason(self.round_report.rejections, message)

    def _count_action(self, kind: str) -> None:
        report = self.round_report
        if kind == "discard" or kind == "hu":
            self._turn_discarded = True
        if report is None:
            return
        if kind == "draw":
            report.draws += 1
        elif kind == "discard":
            report.discards += 1
        elif kind == "hu":
            report.hu_clicked = True

    def _note_eight_card_decision(
        self,
        snapshot: GameSnapshot,
        decision: GameDecision,
        tracker: TableTracker,
    ) -> None:
        """记录能胡/放弃胡牌，以及出牌后是否听牌、听几张。"""
        report = self.round_report
        if report is None:
            return
        if self._hu_offered(snapshot):
            report.hu_available += 1
            if decision.action != "hu":
                report.passed_hu += 1
        if decision.action != "discard" or not decision.card_indexes:
            return
        index = decision.card_indexes[0]
        kept = [card for i, card in enumerate(snapshot.hand) if i != index]
        try:
            unseen = tracker.unseen_counts()
        except RuntimeError:
            return
        waits = sum(
            count
            for key, count in unseen.items()
            if evaluate_hu(kept + [card_from_key(key)], snapshot.gold_ranks) is not None
        )
        if waits > 0:
            report.tenpai_turns += 1
            report.best_waits = max(report.best_waits, waits)

    def round_stats(self, last: int | None = None) -> dict[str, object]:
        return self.round_log.stats(last)

    def _turn_deadline(self) -> float | None:
        """本回合最晚应完成决策的时刻（已扣除点击余量）；未知时返回 None。"""
        if self._turn_started_at is None:
            return None
        return self._turn_started_at + TURN_SECONDS - ACTION_MARGIN_SECONDS

    async def _play_opportunity_card(
        self,
        frame: CapturedFrame,
        opportunity_point: Point | None = None,
    ) -> None:
        """选中检测到的“?”选牌卡并点击“出牌”，随后处理四选一。"""
        def to_frame(point: Point) -> Point:
            return Point(
                round(point.x * frame.width / 1000),
                round(point.y * frame.height / 1000),
            )

        # 兼容旧探针没有回传坐标的情况；新探针会扫描全部 8 个牌位。
        card_point = opportunity_point or to_frame(Point(TURN_HAND_SLOTS[8][0], TURN_HAND_Y))
        discard_point = to_frame(TURN_DISCARD_BUTTON)
        try:
            await self._require_action_state("discard")
            await self._click(frame, card_point)
            await asyncio.sleep(0.30)
            discard_point = await self._require_discard_button()
            await self._click(frame, discard_point)
        except StaleActionError as exc:
            LOGGER.warning("八雀打出选牌卡已过期：%s", exc)
            return
        self._opportunity_attempts += 1
        self.action_count += 1
        self._last_action_at = time.monotonic()
        if self.round_report is not None:
            self.round_report.opportunity_cards += 1
        LOGGER.info(
            "八雀出牌阶段：打出“?”选牌卡 card=(%s,%s) discard=(%s,%s) attempt=%s",
            card_point.x,
            card_point.y,
            discard_point.x,
            discard_point.y,
            self._opportunity_attempts,
        )
        # 等待“请选择想要的牌”弹窗出现。
        await asyncio.sleep(0.6)

    def _decision_tracker(self, snapshot: GameSnapshot) -> TableTracker:
        """给决策用的记牌副本；状态不合法时先修正，修不好就用只含自己手牌的干净副本。"""
        tracker = copy.deepcopy(self.tracker)
        try:
            tracker.unseen_counts(include_opportunity=True)
            return tracker
        except RuntimeError:
            pass
        try:
            tracker.repair_counts()
            return tracker
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning("八雀记牌修正失败，本手用干净记牌器：%s", exc)
        fresh = TableTracker()
        fresh.reset_round(snapshot.hand, snapshot.deck_count)
        return fresh

    @staticmethod
    def _row_hu_flags(
        image_path: Path, cards: list[Card], *, local: bool
    ) -> tuple[bool, ...] | None:
        """亮牌（与 cards 同序）上是否有游戏标的“胡”角标；cards 为 0-1000（local）或像素坐标。"""
        try:
            with Image.open(image_path) as image:
                width = image.width
                badges = detect_table_hu_badges(image)
        except OSError:
            return None
        flags: list[bool] = []
        for card in cards:
            if card.x is None:
                return None
            x = card.x if local else card.x * 1000 / max(1, width)
            slot = min(badges, key=lambda value: abs(value - x))
            flags.append(badges[slot] if abs(slot - x) <= 30 else False)
        return tuple(flags)

    def _log_rule_mismatches(self, snapshot: GameSnapshot) -> None:
        """规则引擎与游戏“胡”角标判定不一致：打日志并存档，用于修正胡牌规则。"""
        mismatches, self.ev.last_rule_mismatches = list(self.ev.last_rule_mismatches), []
        if not mismatches:
            return
        log_path = self.round_log.path
        path = log_path.parent / "rule_mismatches.jsonl" if log_path is not None else None
        for text in mismatches:
            LOGGER.warning("八雀规则核对：%s（金牌 %s）", text, "/".join(snapshot.gold_ranks))
            if path is None:
                continue
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "time": datetime.now().isoformat(timespec="seconds"),
                                "gold": list(snapshot.gold_ranks),
                                "detail": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            except OSError:
                pass

    async def _ensure_round_mode(self) -> None:
        """换牌前确定同花顺优先/追大牌：第一局还没读过底分和豆数时快速读一次（最多 3 秒）。"""
        if self._mode_ready:
            return
        try:
            await asyncio.wait_for(self._read_round_info(), timeout=3.0)
        except asyncio.TimeoutError:
            LOGGER.warning("八雀换牌前读取本局信息超时，按默认评分换牌")

    def _announce_target(self, hand: list[Card]) -> None:
        self._target_pending = False
        gold = self._round_gold_ranks or self._last_gold_ranks
        if not gold:
            return
        LOGGER.info(
            "【本局目标】手牌 [%s] → %s",
            " ".join(card.label for card in hand),
            target_line(self.ev, hand, tuple(gold)),
            extra={"console_red": True},
        )

    async def _announce_target_from_idle(self) -> None:
        """换牌后第一次对手回合：读一次自己的手牌，给出本局主攻方向。"""
        self._target_pending = False
        try:
            frame = await asyncio.to_thread(
                capture_game_window, self.screenshots_dir, self.window_title_keyword
            )
            snapshot = await self.vision.parse_hand(frame.path)
        except (RuntimeError, ValueError, AttributeError, OSError) as exc:
            LOGGER.warning("八雀开局读手牌失败，等轮到自己时再给出本局目标：%s", exc)
            self._target_pending = True
            return
        if len(snapshot.hand) + snapshot.opportunity_card_count != 7:
            self._target_pending = True  # 读得不完整（动画中），轮到自己时再给
            return
        # 对手回合里读到的手牌也作为“已知手牌”：第一次摸牌即可走快速路径、提前算摸牌选项。
        self._known_hand = (list(snapshot.hand), snapshot.opportunity_card_count)
        self._announce_target(snapshot.hand)

    def _locked_hand_if_hu(self, snapshot: GameSnapshot) -> list[Card] | None:
        """现在胡的话会锁定的 7 张：就是上回合出牌后记住的手牌（须包含在当前 8 张里）。"""
        known = self._known_hand
        if known is None or known[1] != 0 or len(known[0]) != 7 or len(snapshot.hand) != 8:
            return None
        remaining = Counter(card_key(card) for card in snapshot.hand)
        remaining.subtract(card_key(card) for card in known[0])
        if any(count < 0 for count in remaining.values()):
            return None
        return list(known[0])

    @staticmethod
    def _draw_key(
        hand: list[Card], opportunity_count: int, table: list[Card], deck: int | None
    ) -> tuple[object, ...]:
        return (
            tuple(card_key(card) for card in hand),
            opportunity_count,
            tuple(card_key(card) for card in table),
            deck,
        )

    def _known_hand_matches(self, frame: CapturedFrame) -> bool:
        """已知手牌（显示顺序）与这一帧的像素花色逐位一致；对不上就丢弃已知手牌。"""
        known = self._known_hand
        if known is None:
            return False
        hand, opportunity_count = known
        if len(hand) + opportunity_count != 7:
            return False
        expected: list[str | None] = [None] * opportunity_count + [
            None if card.wild else card.suit for card in hand
        ]
        try:
            with Image.open(frame.path) as image:
                pixel = classify_hand_suits(image, 7)
        except OSError:
            return False
        if len(pixel) != 7:
            return False
        checked = 0
        for want, got in zip(expected, pixel):
            if got is None:
                continue
            if want != got:
                LOGGER.info("八雀摸牌快速路径：像素花色与已知手牌不符，改用完整识别")
                self._known_hand = None
                return False
            checked += 1
        return checked >= sum(1 for want in expected if want is not None) - 1

    async def _fast_draw_snapshot(self, frame: CapturedFrame) -> GameSnapshot | None:
        """摸牌阶段快速路径：手牌沿用上回合出牌后记住的 7 张（像素核对花色），不再让模型读。

        亮牌行：三家出牌都已看到且之后画面没变 → 直接用对手回合的读数；否则（上家刚出完
        牌就轮到自己，最常见）只读亮牌行（约 0.7 秒），比完整识别省约 1 秒。
        """
        known = self._known_hand
        if known is None or not self._round_gold_ranks:
            return None
        if not await asyncio.to_thread(self._known_hand_matches, frame):
            return None
        hand, opportunity_count = known
        row = self._last_row_local
        reuse = (
            row is not None
            and row[1] is not None
            and self._row_diff.cycle_complete
            and not table_signature_changed(self._row_prev_signature, self._row_read_signature)
        )
        if reuse:
            assert row is not None
            cards, deck = row
            source = "对手回合读数"
        else:
            try:
                cards, deck = await self.vision.parse_table(frame.path)
            except (RuntimeError, ValueError, AttributeError, OSError) as exc:
                LOGGER.warning("八雀摸牌快速路径读亮牌失败，改用完整识别：%s", exc)
                return None
            if deck is None:
                return None
            self._last_row_local = (list(cards), deck)
            source = "只读亮牌行"
        snapshot = self.vision.build_turn_snapshot(
            hand=hand,
            opportunity_count=opportunity_count,
            table=cards,
            deck_count=deck,
            gold_ranks=self._round_gold_ranks,
            action_state="draw",
            size=(frame.width, frame.height),
        )
        snapshot.visible_hu = self._row_hu_flags(frame.path, snapshot.visible_cards, local=False)
        LOGGER.info(
            "八雀摸牌快速路径（%s）：沿用已知手牌 [%s]，亮牌行 [%s]",
            source,
            " ".join(card.label for card in hand),
            " ".join(card.label for card in cards) or "空",
        )
        return snapshot

    def _cancel_draw_precompute(self) -> None:
        if self._draw_precompute is not None:
            self._draw_precompute[1].cancel()
            self._draw_precompute = None

    def _start_draw_precompute(self) -> None:
        """上家出完牌、亮牌行稳定：趁对手回合空闲先把摸牌决策算好。"""
        known, row = self._known_hand, self._last_row_local
        if (
            self.decision_mode == "rule"
            or known is None
            or row is None
            or not self._round_gold_ranks
            or not self._tracker_started
        ):
            return
        hand, opportunity_count = known
        cards, deck = row
        key = (*self._draw_key(hand, opportunity_count, cards, deck), self._last_row_hu)
        if self._draw_precompute is not None:
            if self._draw_precompute[0] == key or not self._draw_precompute[1].done():
                # 同一局面已在算，或上一次还没算完（线程无法中途取消，别再叠一个抢 CPU）。
                return
        tracker = copy.deepcopy(self.tracker)
        try:
            tracker.sync_table(cards)
            tracker.unseen_counts(include_opportunity=True)
        except (RuntimeError, ValueError):
            try:
                tracker.repair_counts()
            except (RuntimeError, ValueError):
                return
        future = asyncio.ensure_future(
            asyncio.to_thread(
                self.ev.choose_draw_action,
                list(hand),
                list(cards),
                self._round_gold_ranks,
                tracker,
                opportunity_count=opportunity_count,
                time_budget_seconds=self.decision_budget_seconds,
                visible_hu=self._last_row_hu,
            )
        )
        self._draw_precompute = (key, future)

    def _start_discard_precompute(self, hand7: list[Card], taken: Card) -> None:
        """拿亮牌后 8 张已确定：点击的同时在后台算出牌决策，出牌阶段识别完直接用。"""
        self._discard_precompute = None
        if self.decision_mode == "rule" or not self._round_gold_ranks or not self._tracker_started:
            return
        hand8 = hand7 + [taken]
        gold = self._round_gold_ranks
        hu_allowed = evaluate_hu(hand8, gold) is not None
        tracker = copy.deepcopy(self.tracker)
        try:
            tracker.update_own_hand(hand8)
        except (RuntimeError, ValueError):
            try:
                tracker.repair_counts()
            except (RuntimeError, ValueError):
                return
        future = asyncio.ensure_future(
            asyncio.to_thread(
                self.ev.choose_8_card_action,
                hand8,
                gold,
                tracker,
                time_budget_seconds=self.decision_budget_seconds,
                avoid_discard=taken,
                hu_allowed=hu_allowed,
                locked_hand=list(hand7),
                self_draw=False,
            )
        )
        key = tuple(sorted(card_key(card) for card in hand8))
        self._discard_precompute = (key, hu_allowed, hand8, future)

    async def _take_discard_precompute(self, snapshot: GameSnapshot) -> GameDecision | None:
        """出牌阶段：8 张与拿牌时一致、能否胡也一致，就用拿牌时算好的出牌决策（按牌对应位置）。"""
        pending, self._discard_precompute = self._discard_precompute, None
        if pending is None or snapshot.screen_type not in {ScreenType.DISCARD, ScreenType.HU}:
            return None
        key, hu_allowed, hand8, future = pending
        if (
            tuple(sorted(card_key(card) for card in snapshot.hand)) != key
            or hu_allowed != self._hu_offered(snapshot)
        ):
            return None
        try:
            decision, values = await future
        except (asyncio.CancelledError, RuntimeError, ValueError) as exc:
            LOGGER.warning("八雀预算出牌决策不可用，当场重算：%s", exc)
            return None
        if decision.action == "discard":
            wanted = card_key(hand8[decision.card_indexes[0]])
            matches = [i for i, card in enumerate(snapshot.hand) if card_key(card) == wanted]
            if not matches:
                return None
            unselected = [i for i in matches if not snapshot.hand[i].selected]
            decision = replace(decision, card_indexes=((unselected or matches)[0],))
        self._note_eight_card_decision(snapshot, decision, self.tracker)
        self.last_decision_source = "ev_precomputed"
        LOGGER.info("八雀出牌决策：使用拿牌时预先算好的结果")
        return decision

    async def _take_draw_precompute(self, snapshot: GameSnapshot) -> GameDecision | None:
        """局面与预算时一致（手牌、亮牌行、牌堆数）就直接用预先算好的摸牌决策。"""
        pending, self._draw_precompute = self._draw_precompute, None
        if pending is None:
            return None
        key, future = pending
        if not future.done():
            # 后台预算还在跑：等它一小会儿（它在填逐选项缓存），免得和当场计算抢 CPU。
            try:
                await asyncio.wait_for(asyncio.shield(future), timeout=0.5)
            except (asyncio.TimeoutError, asyncio.CancelledError, RuntimeError, ValueError):
                pass
        if snapshot.screen_type != ScreenType.DRAW or key != (
            *self._draw_key(
                snapshot.hand,
                snapshot.opportunity_card_count,
                snapshot.visible_cards,
                snapshot.deck_count,
            ),
            snapshot.visible_hu,
        ):
            future.cancel()
            return None
        try:
            decision = await future
        except (asyncio.CancelledError, RuntimeError, ValueError) as exc:
            LOGGER.warning("八雀预算摸牌决策不可用，当场重算：%s", exc)
            return None
        self.last_decision_source = "ev_precomputed"
        LOGGER.info("八雀摸牌决策：使用对手回合预先算好的结果")
        return decision

    def _start_row_watch(self, discarded: Card) -> None:
        """自己出完牌：开始按亮牌行变化记录下家、对家、上家的动作。"""
        if self._own_draw_row is None:
            # 不知道本回合摸牌前的亮牌行（如开局第一手），这一轮无法可靠归属。
            self._row_diff.reset()
            return
        row, deck, taken, drew_deck = self._own_draw_row
        self._own_draw_row = None
        self._row_diff.start_after_own_turn(
            row, deck, taken=taken, discarded=discarded, drew_deck=drew_deck
        )
        self._row_prev_signature = None
        self._row_read_signature = None

    async def _watch_table(self, probe: TurnProbe) -> None:
        """对手回合：亮牌区画面变化并稳定后读一次亮牌行，归属到对应座位。"""
        signature = probe.table_signature
        if not self._row_diff.active or signature is None:
            return
        stable = not table_signature_changed(signature, self._row_prev_signature)
        self._row_prev_signature = signature
        if not stable or not table_signature_changed(signature, self._row_read_signature):
            return
        self._row_read_signature = signature
        try:
            frame = await asyncio.to_thread(
                capture_game_window, self.screenshots_dir, self.window_title_keyword
            )
            cards, deck = await self.vision.parse_table(frame.path)
        except (RuntimeError, ValueError, AttributeError, OSError) as exc:
            LOGGER.warning("八雀对手回合亮牌识别失败：%s", exc)
            return
        self._last_row_local = (list(cards), deck)
        self._last_row_hu = self._row_hu_flags(frame.path, cards, local=True)
        self._apply_row_read(cards, deck)
        # 不等三家出完（上家一出完就轮到自己，来不及）：每次亮牌行变化都先把现有选项算好，
        # 轮到自己时只需补算上家新打出的那张（见 EV 的逐选项缓存）。
        self._start_draw_precompute()

    def _on_settlement(self, winner: str) -> None:
        """胡牌结算：记下赢家；对手拿亮牌胡时把那张牌记进他手里，并停止亮牌归属。

        结算时亮牌行会翻出别的牌，不能再当成出牌。有人胡后本局可能继续，
        所以只重置亮牌比对，等自己下一次出牌后重新开始，不结束本局。
        """
        if self._settlement_active:
            return
        self._settlement_active = True
        # 结算后各家豆数都变了：下一个对手回合重新读本局信息，更新封顶。
        self._round_info_attempts = 0
        names = {"right": "下家", "top": "对家", "left": "上家", "self": "自己"}
        # 胡了之后轮到赢家的下家摸牌，赢家照常留在牌局里：从下家接着归属。
        taken = self._row_diff.continue_after_hu(winner)
        if winner in names and winner != "self" and taken is not None and self._tracker_started:
            try:
                self.tracker.record_opponent_take_visible(winner, taken)
                LOGGER.info("写入对手事件 %s 拿亮牌 %s → 胡牌", winner, taken.label)
                self.opponent_event_count += 1
            except (RuntimeError, ValueError) as exc:
                LOGGER.warning("八雀胡牌拿牌写入记牌失败：%s", exc)
                try:
                    self.tracker.repair_counts()
                except (RuntimeError, ValueError):
                    pass
        self._observer_recent_events.append(f"{winner} 胡牌")
        LOGGER.info(
            "八雀检测到胡牌结算：%s胡牌%s",
            names.get(winner, "未知玩家"),
            f"（拿亮牌 {taken.label}）" if taken is not None and winner != "self" else "",
            extra={"console_red": True},
        )

    def _apply_row_read(self, cards: list[Card], deck: int | None) -> None:
        LOGGER.info(
            "八雀亮牌行读数 牌堆=%s [%s]", deck, " ".join(card.label for card in cards) or "空"
        )
        for turn in self._row_diff.observe(list(cards), deck):
            self._record_opponent_turn(turn)

    def _record_opponent_turn(self, turn: OpponentTurn) -> None:
        if not self._tracker_started:
            return
        try:
            self.tracker.record_opponent_turn(
                turn.seat,
                draw_source=turn.draw_source,
                discard=turn.discard,
                drawn_card=turn.drawn_card,
            )
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning("八雀对手事件写入记牌失败（%s）：%s", turn.describe(), exc)
            try:
                self.tracker.repair_counts()  # 写入可能做了一半，修回非负计数
            except (RuntimeError, ValueError):
                pass
            return
        self.opponent_event_count += 1
        self._observer_recent_events.append(turn.describe())
        LOGGER.info("写入对手事件 %s", turn.describe())

    async def _handle_choice_popup(self, frame: CapturedFrame) -> None:
        """使用“?”选牌卡后会弹出“请选择想要的牌”，必须在倒计时内点选一张。"""
        started_at = time.monotonic()
        try:
            options, hand = await self.vision.parse_choice(frame.path)
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning("八雀选牌弹窗识别失败，重新探测：%s", exc)
            return
        if not options:
            LOGGER.warning("八雀选牌弹窗未识别到候选牌，重新探测")
            return
        # 开局 ?+6 张时记牌器尚未建立，金牌退回用最近一帧识别结果。
        gold_ranks = self._round_gold_ranks or (
            self.last_snapshot.gold_ranks if self.last_snapshot is not None else ()
        )
        # 牌堆顶风险要对三家做采样：放到工作线程并限时，弹窗倒计时很短。
        tracker = copy.deepcopy(self.tracker) if self._tracker_started else None
        index, value = await asyncio.to_thread(
            self.ev.choose_option_card,
            hand,
            options,
            gold_ranks,
            tracker,
            time_budget_seconds=CHOICE_RISK_SECONDS,
        )
        card = options[index]
        if card.x is None or card.y is None or not (
            0 <= card.x < frame.width and 0 <= card.y < frame.height
        ):
            LOGGER.warning("八雀选牌弹窗候选坐标异常：%s (%s,%s)", card.label, card.x, card.y)
            return
        await self._click(frame, Point(card.x, card.y))
        self.action_count += 1
        self._last_action_at = time.monotonic()
        if self._tracker_started and len(options) == 4:
            try:
                # 未选中的三张按原顺序回到牌堆顶，之后摸牌时可确知。
                self.tracker.record_opportunity_reveal("self", options, index)
            except (ValueError, IndexError) as exc:
                LOGGER.warning("八雀选牌结果写入记牌器失败：%s", exc)
        if self.round_report is not None:
            self.round_report.choice_picks += 1
        LOGGER.info(
            "八雀选牌弹窗：选择 %s（价值 %.1f）候选=%s 手牌=%s duration=%.2fs｜%s",
            card.label,
            value,
            [option.label for option in options],
            [item.label for item in hand],
            time.monotonic() - started_at,
            self.ev.last_option_notes,
        )
        # 等弹窗关闭、牌进入手牌后再继续探测出牌阶段。
        await asyncio.sleep(0.6)

    async def _start_page_ok(
        self,
        frame: CapturedFrame,
        point: Point,
        *,
        live: bool = True,
    ) -> bool:
        """点击“开始游戏”前的安全闸：识别所用截图与（可选）即时画面都必须像开始页。"""
        try:
            with Image.open(frame.path) as image:
                frame_ok = is_start_page_button(image.convert("RGB"), point)
        except OSError:
            frame_ok = False
        live_ok = True
        if frame_ok and live:
            try:
                live_ok = await asyncio.to_thread(
                    current_start_page_check,
                    self.window_title_keyword,
                    point,
                )
            except RuntimeError:
                live_ok = False
        if not (frame_ok and live_ok):
            LOGGER.warning(
                "拒绝点击疑似“开始游戏”按钮 (%s,%s)：不像开始页（可能是购买/充值弹窗或牌局中的按钮）",
                point.x,
                point.y,
            )
            return False
        return True

    async def _refocus_game(self) -> None:
        """微信消息弹窗等会抢走前台；重新切回游戏，失败只记录不中断。"""
        try:
            await asyncio.to_thread(focus_game_window, self.window_title_keyword)
        except Exception as exc:  # pywinauto 在窗口切换中可能抛出多种异常
            LOGGER.warning("重新切回游戏窗口失败：%s", exc)
        await asyncio.sleep(0.30)

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

        if (
            snapshot.screen_type == ScreenType.START
            and self._turn_started_at is not None
            and time.monotonic() - self._turn_started_at < TURN_SECONDS + 5
        ):
            # 回合进行中不可能出现开始页；OpenCV 曾把“出牌”按钮误判为“开始游戏”。
            return GameStepResult(
                False,
                "回合进行中，忽略疑似“开始游戏”按钮（多半是出牌按钮）。",
                snapshot,
                frame,
            )

        if snapshot.screen_type == ScreenType.START:
            if self.round_report is not None and self.round_report.turns > 0:
                self._finish_round_report("回到开始页")
            if snapshot.start_button is None:
                return GameStepResult(
                    False,
                    "已识别到八雀开始页面，但没有可靠识别到“开始游戏”按钮，暂不点击。",
                    snapshot,
                    frame,
                )
            if execute:
                if not await self._start_page_ok(frame, snapshot.start_button):
                    return GameStepResult(
                        False,
                        "疑似“开始游戏”按钮未通过安全校验（可能是购买弹窗），未点击。",
                        snapshot,
                        frame,
                    )
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
            self._finish_round_report("豆子不足/充值界面")
            return GameStepResult(
                False,
                "检测到“豆子不足/充值/购买”界面。自动游戏不会充值或购买，已停止。",
                snapshot,
                frame,
                stop_loop=True,
            )
        if snapshot.screen_type == ScreenType.LOBBY:
            self._finish_round_report("返回场次选择页")
            return GameStepResult(
                False,
                "检测到八雀场次选择页，本次游戏已经终止，自动游戏已停止。",
                snapshot,
                frame,
                stop_loop=True,
            )
        if snapshot.screen_type == ScreenType.SETTLEMENT:
            self._finish_round_report("结算页")
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
            self._note_turn_problem(validation_error, rejection=True)
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
                self._note_turn_problem("画面过期")
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
        tracker_snapshot = self._decision_tracker(snapshot)
        started_at = time.monotonic()
        precomputed = (
            await self._take_draw_precompute(snapshot)
            if snapshot.screen_type == ScreenType.DRAW
            else await self._take_discard_precompute(snapshot)
        )
        try:
            if precomputed is not None:
                decision = precomputed
            else:
                decision = await asyncio.to_thread(
                    self._decide,
                    snapshot,
                    tracker_snapshot,
                    self._turn_deadline(),
                )
        except RuntimeError as exc:
            if "记牌" not in str(exc):
                raise
            # 记牌器损坏绝不能让整局自动游戏退出：这一手改用只含自己手牌的干净记牌器。
            LOGGER.warning("八雀决策时记牌异常，本手改用干净记牌器：%s", exc)
            fresh = TableTracker()
            fresh.reset_round(snapshot.hand, snapshot.deck_count)
            decision = await asyncio.to_thread(self._decide, snapshot, fresh, self._turn_deadline())
        if snapshot.screen_type == ScreenType.DRAW:
            self._log_rule_mismatches(snapshot)
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
            self._note_turn_problem(str(exc))
            return GameStepResult(False, str(exc), snapshot, frame)
        self.action_count += 1
        self._last_action_fingerprint = fingerprint
        self._last_action_at = time.monotonic()
        if decision.action == "discard" and decision.card_indexes and len(snapshot.hand) == 8:
            await self._log_plan(snapshot, decision, tracker_snapshot)
        return GameStepResult(True, message, snapshot, frame)

    async def _log_plan(
        self,
        snapshot: GameSnapshot,
        decision: GameDecision,
        tracker: TableTracker,
    ) -> None:
        """出牌后红色播报：自己准备做什么牌型，上家/对家/下家可能在做什么牌型。"""
        index = decision.card_indexes[0]
        kept = [card for i, card in enumerate(snapshot.hand) if i != index]
        gold = tuple(snapshot.gold_ranks)

        def build() -> list[str]:
            lines = [
                "目标进展：" + target_line(self.ev, kept, gold),
                own_plan_line(
                    self.ev, kept, gold, tracker, deadline=time.monotonic() + PLAN_LOG_SECONDS
                ),
            ]
            lines.extend(
                opponent_plan_lines(
                    self.ev, gold, tracker, deadline=time.monotonic() + PLAN_LOG_SECONDS
                )
            )
            return lines

        try:
            lines = await asyncio.to_thread(build)
        except Exception as exc:  # 播报只是日志，任何失败都不能影响对局
            LOGGER.warning("八雀牌型计划播报失败：%s", exc)
            return
        for line in lines:
            LOGGER.info("【牌型计划】%s", line, extra={"console_red": True})

    async def prepare_auto_round(self, timeout_seconds: float = 30.0) -> GameSnapshot:
        """自动模式启动前，把游戏切到前台并从“开始游戏”页面进入正式牌局。

        只有识别到真实“开始游戏”按钮才点击，不使用固定坐标；进入换牌 6 张
        或后续 7/8 张手牌
        和 3 个金牌的正式牌局后才返回，随后插件再启动三家观察器。
        """
        self._swap_phase_active = True
        self._start_round_report("自动开局")
        await asyncio.to_thread(focus_game_window, self.window_title_keyword)
        # Windows 前台切换存在极短的异步窗口，给系统一点时间完成激活。
        await asyncio.sleep(0.30)
        LOGGER.info("八雀启动流程：已切换游戏窗口到前台")

        deadline = time.monotonic() + max(5.0, float(timeout_seconds))
        start_click_attempts = 0
        last_start_click_at = 0.0
        focus_retries = 0
        prefetched: tuple[CapturedFrame, GameSnapshot] | None = None
        while time.monotonic() < deadline:
            # 自动启动只关心当前处于开始、匹配、发牌还是正式牌局；历史帧和
            # 对手事件此时没有价值，双图反而会让本地视觉模型显著变慢。
            if prefetched is not None:
                frame, snapshot = prefetched
                prefetched = None
            else:
                try:
                    frame, snapshot = await self.inspect(include_previous=False)
                except RuntimeError as exc:
                    # 切到游戏后，微信等窗口可能立刻又抢回前台。
                    if not _is_foreground_error(exc) or focus_retries >= 5:
                        raise
                    focus_retries += 1
                    LOGGER.warning(
                        "八雀启动流程：游戏窗口不在前台，重新切回 attempt=%s：%s",
                        focus_retries,
                        exc,
                    )
                    await self._refocus_game()
                    continue

            if snapshot.screen_type == ScreenType.RECHARGE:
                raise RuntimeError("检测到豆子不足/充值/购买界面，自动游戏不会继续。")
            if snapshot.screen_type == ScreenType.LOBBY:
                raise RuntimeError("检测到八雀场次选择页，本次游戏已经终止。")

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
                    if not await self._start_page_ok(frame, start_button):
                        await asyncio.sleep(0.8)
                        continue
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
                                self._hand_slot_count(swap_snapshot) == 7
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
                    (
                        snapshot.screen_type == ScreenType.SWAP
                        and self._hand_slot_count(snapshot) == 7
                    )
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
                        await self._ensure_round_mode()
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
                    if _is_foreground_error(exc):
                        # 回合倒计时很短，被其他窗口抢走前台时立即切回，而不是
                        # 干等用户手动切换（与操作前复核 _current_action_state 一致）。
                        LOGGER.warning("游戏窗口不在前台，重新切回后重试：%s", exc)
                        await self._refocus_game()
                        if "没有找到包含" in str(exc):
                            # 游戏窗口根本不存在时不要高频空转。
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
            self._finish_round_report("停止自动游戏")
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
        deadline: float | None = None,
    ) -> GameDecision:
        """deadline 为本回合最晚决策时刻（time.monotonic）；EV 预算与是否调用
        本地重排模型都以它为上限，保证在倒计时结束前点出动作。"""
        tracker = tracker or self.tracker
        budget = self.decision_budget_seconds
        if deadline is not None:
            budget = min(budget, max(0.5, deadline - time.monotonic()))

        if snapshot.screen_type == ScreenType.SWAP:
            self.last_decision_source = "rule"
            # 与出牌同一套标准：同花顺优先 / 追大牌（由场次与自己豆数决定）。
            return choose_swap_two(
                snapshot.hand,
                snapshot.gold_ranks,
                prefer_flush=self.ev.prefer_flush,
                chase_big=self.ev.chase_big,
                prefer_pinghu=self.ev.prefer_pinghu,
            )

        if self.decision_mode == "rule":
            self.last_decision_source = "rule"
            if snapshot.screen_type == ScreenType.DRAW:
                if snapshot.opportunity_card_count:
                    # 规则策略只支持 7 张普通手牌；含问号牌时复用可处理 5/6 张
                    # 真实手牌的轻量 EV 路径，避免双问号牌局卡住。
                    self.last_decision_source = "ev"
                    return self.ev.choose_draw_action(
                        snapshot.hand,
                        snapshot.visible_cards,
                        snapshot.gold_ranks,
                        tracker,
                        opportunity_count=snapshot.opportunity_card_count,
                        time_budget_seconds=budget,
                        visible_hu=snapshot.visible_hu,
                    )
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
                opportunity_count=snapshot.opportunity_card_count,
                time_budget_seconds=budget,
                visible_hu=snapshot.visible_hu,
            )

        if snapshot.screen_type in {ScreenType.DISCARD, ScreenType.HU}:
            decision, values = self.ev.choose_8_card_action(
                snapshot.hand,
                snapshot.gold_ranks,
                tracker,
                time_budget_seconds=budget,
                avoid_discard=self._taken_visible,
                hu_allowed=self._hu_offered(snapshot),
                locked_hand=self._locked_hand_if_hu(snapshot),
                # 本回合没拿亮牌 = 这张是从牌堆摸来的，胡了算自摸（×2）。
                self_draw=self._taken_visible is None,
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

            time_left = None if deadline is None else deadline - time.monotonic()
            if decision.action == "hu":
                # 确定能胡且 EV 选择胡牌时直接胡，不再让本地模型重排。
                self.last_decision_source = "ev"
            elif (
                self.decision_mode == "hybrid"
                and self.decision_reranker is not None
                and time_left is not None
                and time_left < self.decision_reranker.timeout_seconds + HYBRID_SPARE_SECONDS
            ):
                # 本地模型最坏要等满超时；剩余时间不够时直接用 EV，不能拿回合冒险。
                self.hybrid_fallback_count += 1
                self.last_decision_source = "hybrid_ev_fallback"
                LOGGER.info(
                    "八雀HYBRID跳过本地模型：回合剩余 %.1fs < 模型超时 %.1fs，沿用Python EV",
                    time_left,
                    self.decision_reranker.timeout_seconds,
                )
            elif self.decision_mode == "hybrid" and self.decision_reranker is not None:
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

            self._note_eight_card_decision(snapshot, decision, tracker)
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
            self._record(self.tracker.record_own_take_visible, card)
            self._taken_visible = card
            self._start_discard_precompute(list(snapshot.hand), card)
            self._own_draw_row = (list(snapshot.visible_cards), snapshot.deck_count, card, False)
            self._count_action("draw")
            return decision.reason + "；已取亮牌。"

        if decision.action == "draw_deck":
            await self._require_action_state("draw")
            if snapshot.deck_click is None:
                raise RuntimeError("未识别到牌堆点击位置")
            await self._click(frame, snapshot.deck_click)
            self._record(self.tracker.record_own_draw_deck)
            self._own_draw_row = (list(snapshot.visible_cards), snapshot.deck_count, None, True)
            self._count_action("draw")
            return decision.reason + "；已摸牌堆。"

        if decision.action == "hu":
            await self._require_action_state("discard")
            live_hu = (await self._read_action_buttons()).get("hu")
            if live_hu is not None:
                await self._click(frame, live_hu)
                self._count_action("hu")
                return decision.reason + "；已点击胡牌。"
            if snapshot.hu_button is None:
                raise RuntimeError(
                    f"规则引擎判断已胡（{decision.hu.pattern if decision.hu else '未知'}），"
                    "但视觉没有可靠识别到胡牌按钮；为避免把胡牌打掉，本轮不操作。"
                )
            await self._click(frame, snapshot.hu_button)
            self._count_action("hu")
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
            # 用旧截图里的“出牌”按钮坐标点击。选牌后蓝色横幅会消失，
            # 这里只能看“出牌”按钮是否还在。
            discard_point = await self._require_discard_button()
            await self._click(frame, discard_point)
            self._record(self.tracker.record_own_discard, desired)
            self._known_hand = (
                [card for i, card in enumerate(snapshot.hand) if i != index],
                snapshot.opportunity_card_count,
            )
            self._start_row_watch(desired)
            self._count_action("discard")
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

    async def _read_action_buttons(self) -> dict[str, Point]:
        for attempt in range(2):
            try:
                return await asyncio.to_thread(current_action_buttons, self.window_title_keyword)
            except RuntimeError as exc:
                LOGGER.warning(
                    "读取操作按钮失败，尝试重新聚焦 attempt=%s error=%s",
                    attempt + 1,
                    exc,
                )
                await asyncio.to_thread(focus_game_window, self.window_title_keyword)
                await asyncio.sleep(0.15)
        return {}

    async def _require_discard_button(self) -> Point:
        """确认“出牌”按钮仍在并返回其即时位置（能胡时它会左移）。"""
        point = (await self._read_action_buttons()).get("discard")
        if point is None:
            # 保存当时画面，便于排查是回合真结束还是出现了新的界面元素。
            try:
                frame = await asyncio.to_thread(
                    capture_game_window,
                    self.screenshots_dir,
                    self.window_title_keyword,
                )
                LOGGER.warning("八雀出牌按钮复核失败，现场截图：%s", frame.path)
            except RuntimeError:
                pass
            raise StaleActionError("选牌后复核失败：“出牌”按钮已消失，回合已结束，已取消点击。")
        return point

    async def _require_action_state(self, expected: str) -> None:
        actual = await self._current_action_state()
        deadline = time.monotonic() + STATE_RECHECK_SECONDS
        while actual != expected and time.monotonic() < deadline:
            # 表情动画、提示面板飘过时横幅会短暂消失；稍等再看，避免白白丢掉回合。
            await asyncio.sleep(0.3)
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

    def _stabilize_gold_ranks(self, snapshot: GameSnapshot) -> None:
        """牌局中途金牌读数与本局不符时，先当作误读，用本局金牌覆盖快照。

        金牌整局不变，变化只可能是新一局：牌堆接近满堆，或连续多次读到同一组新金牌。
        否则一次 8→6 误读就会清空三家记牌，且让 EV 把真金牌当普通牌打出。
        """
        if not self._round_gold_ranks or len(snapshot.gold_ranks) != 3:
            return
        new_gold = tuple(snapshot.gold_ranks)
        if new_gold == self._round_gold_ranks:
            self._pending_gold_ranks = ()
            self._pending_gold_hits = 0
            return
        if snapshot.deck_count is not None and snapshot.deck_count >= FRESH_ROUND_DECK_COUNT:
            return
        if new_gold == self._pending_gold_ranks:
            self._pending_gold_hits += 1
        else:
            self._pending_gold_ranks = new_gold
            self._pending_gold_hits = 1
        if self._pending_gold_hits >= GOLD_CHANGE_CONFIRMATIONS:
            return
        LOGGER.warning(
            "八雀金牌读数 %s 与本局 %s 不符（牌堆 %s），暂按误读处理，沿用本局金牌",
            new_gold,
            self._round_gold_ranks,
            snapshot.deck_count,
        )
        snapshot.gold_ranks = self._round_gold_ranks

    def _sync_tracker(self, snapshot: GameSnapshot) -> None:
        self._stabilize_gold_ranks(snapshot)
        if len(snapshot.gold_ranks) == 3:
            self._last_gold_ranks = tuple(snapshot.gold_ranks)
        try:
            self._sync_tracker_unchecked(snapshot)
        except (RuntimeError, ValueError) as exc:
            # 视觉偶尔认错花色（如 10♠→10♣）会让记牌计数为负。先只修正冲突的
            # 那张牌、保留三家记录；修不好再以当前手牌整盘重建。
            try:
                repaired = self.tracker.repair_counts()
                self._sync_tracker_unchecked(snapshot)
                LOGGER.warning(
                    "八雀记牌状态冲突（%s），已修正 %s，保留其余记牌", exc, "、".join(repaired) or "无"
                )
                return
            except (RuntimeError, ValueError):
                pass
            LOGGER.warning("八雀记牌状态冲突，按当前手牌重建：%s", exc)
            if len(snapshot.hand) in {7, 8}:
                try:
                    self.tracker.reset_round(snapshot.hand, snapshot.deck_count)
                    self._tracker_started = True
                except (RuntimeError, ValueError) as reset_exc:
                    LOGGER.warning("八雀记牌重建失败，暂停记牌：%s", reset_exc)
                    self._tracker_started = False

    def _record(self, action: object, *args: object) -> None:
        """自己动作写入记牌器；记牌冲突只记录，不能让已完成的点击变成异常。"""
        try:
            action(*args)  # type: ignore[operator]
        except (RuntimeError, ValueError) as exc:
            LOGGER.warning("八雀记牌写入失败（动作已执行）：%s", exc)

    def _sync_tracker_unchecked(self, snapshot: GameSnapshot) -> None:
        # 每张“?”选牌卡各占一个手牌位；2 张问号 + 5 张真实牌也是完整 7 张。
        hand_slots = self._hand_slot_count(snapshot)
        if len(snapshot.gold_ranks) != 3 or hand_slots not in {7, 8}:
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
            self._pending_gold_ranks = ()
            self._pending_gold_hits = 0
            # 新一局：谁先出牌是随机的，开局就开始记录对手动作（轮到自己时倒推座位）。
            self._row_diff.start_opening()
            self._target_pending = True
            self._own_draw_row = None
            self._known_hand = None
            self._last_row_local = None
            self._cancel_draw_precompute()
            self._pair_event_dedupe.clear()
            self._observation_dedupe.clear()
            self._observer_recent_events.clear()
            self.opponent_event_count = 0
            self._round_info_attempts = 0
            LOGGER.info("建立新一局记牌状态 gold=%s deck=%s", new_gold, snapshot.deck_count)
        else:
            self.tracker.update_own_hand(snapshot.hand)
            # 出牌阶段的快速识别不读牌堆数；没读到时保留已知值。
            if snapshot.deck_count is not None:
                self.tracker.set_deck_count(snapshot.deck_count)
        if snapshot.screen_type == ScreenType.DRAW:
            # 摸牌阶段看得到桌面亮牌（含对手刚打出的牌），这些牌不在牌堆里。
            self.tracker.sync_table(snapshot.visible_cards)

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

    def set_opponent_observer_controls(
        self,
        *,
        pause: Callable[[], Awaitable[None]] | None,
        resume: Callable[[], Awaitable[None]] | None,
    ) -> None:
        self._pause_opponent_observer = pause
        self._resume_opponent_observer = resume
        self._observer_phase = "inactive"

    async def _set_observer_phase(self, phase: str) -> None:
        if phase == self._observer_phase:
            return
        self._observer_phase = phase
        if phase == "opponent":
            callback = self._resume_opponent_observer
        else:
            callback = self._pause_opponent_observer
        if callback is not None:
            await callback()

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
            ScreenType.DRAW: 7 - snapshot.opportunity_card_count,
            ScreenType.DISCARD: 8,
            ScreenType.HU: 8,
        }.get(snapshot.screen_type)
        if (
            snapshot.screen_type == ScreenType.SWAP
            and self._hand_slot_count(snapshot) != 7
        ):
            return (
                "swap 阶段手牌位应为 7 个，当前识别到 "
                f"{len(snapshot.hand)} 张真实牌 + {snapshot.opportunity_card_count} 张问号牌，暂不操作。"
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
        opportunity = str(snapshot.opportunity_card_count) if snapshot.opportunity_card_count else "无"
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

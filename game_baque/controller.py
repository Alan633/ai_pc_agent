from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from agent.session import AgentSession
from game_baque.capture import CapturedFrame, assert_game_foreground, capture_game_window
from game_baque.ev import ExpectedValueEngine
from game_baque.models import Card, GameDecision, GameSnapshot, Point, ScreenType
from game_baque.strategy import choose_swap_two
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
    ) -> None:
        self.vision = vision
        self.screenshots_dir = screenshots_dir
        self.session = session
        self.window_title_keyword = window_title_keyword
        self.action_interval_seconds = max(0.5, float(action_interval_seconds))
        self.min_card_confidence = min(max(float(min_card_confidence), 0.0), 1.0)
        self.min_event_confidence = min(max(float(min_event_confidence), 0.0), 1.0)
        self.auto_next_round = bool(auto_next_round)
        self.last_snapshot: GameSnapshot | None = None
        self.last_message = "尚未识别"
        self.action_count = 0
        self.tracker = TableTracker()
        self.ev = ExpectedValueEngine()
        self._tracker_started = False
        self._round_gold_ranks: tuple[str, ...] = ()
        self._last_action_fingerprint: tuple[object, ...] | None = None
        self._last_action_at = 0.0
        self._previous_frame_path: Path | None = None
        self._recent_opponent_events: dict[tuple[str, str, str], float] = {}
        self.opponent_event_count = 0

    async def inspect(self) -> tuple[CapturedFrame, GameSnapshot]:
        frame = await asyncio.to_thread(
            capture_game_window,
            self.screenshots_dir,
            self.window_title_keyword,
        )
        snapshot = await self.vision.parse(
            frame.path,
            previous_image_path=self._previous_frame_path,
        )
        self._previous_frame_path = frame.path
        self._sync_tracker(snapshot)
        self._apply_opponent_events(snapshot)
        # 当前截图里的 deck_count 是视觉观察到的最终状态；
        # 对手暗摸事件会先按事件语义减 1，这里再用观察值校准，避免双重扣减。
        if snapshot.deck_count is not None and self._tracker_started:
            self.tracker.set_deck_count(snapshot.deck_count)
        self.last_snapshot = snapshot
        self.last_message = self._snapshot_summary(snapshot)
        return frame, snapshot

    async def step(self, *, execute: bool = True) -> GameStepResult:
        frame, snapshot = await self.inspect()

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
            return GameStepResult(False, validation_error, snapshot, frame)

        fingerprint = snapshot.fingerprint()
        if (
            execute
            and fingerprint == self._last_action_fingerprint
            and time.monotonic() - self._last_action_at < max(3.0, self.action_interval_seconds * 1.5)
        ):
            return GameStepResult(False, "画面尚未变化，等待游戏响应。", snapshot, frame)

        decision = self._decide(snapshot)
        if not execute:
            return GameStepResult(False, decision.reason, snapshot, frame)

        message = await self._execute(frame, snapshot, decision)
        self.action_count += 1
        self._last_action_fingerprint = fingerprint
        self._last_action_at = time.monotonic()
        return GameStepResult(True, message, snapshot, frame)

    async def run_loop(self) -> str:
        unknown_streak = 0
        try:
            while not self.session.stop_event.is_set():
                result = await self.step(execute=True)
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
                await asyncio.sleep(self.action_interval_seconds)
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

    def _decide(self, snapshot: GameSnapshot) -> GameDecision:
        if snapshot.screen_type == ScreenType.SWAP:
            return choose_swap_two(snapshot.hand, snapshot.gold_ranks)
        if snapshot.screen_type == ScreenType.DRAW:
            return self.ev.choose_draw_action(
                snapshot.hand,
                snapshot.visible_cards,
                snapshot.gold_ranks,
                self.tracker,
            )
        if snapshot.screen_type in {ScreenType.DISCARD, ScreenType.HU}:
            decision, values = self.ev.choose_8_card_action(
                snapshot.hand,
                snapshot.gold_ranks,
                self.tracker,
            )
            LOGGER.info(
                "八雀EV候选 %s",
                [
                    {
                        "action": item.action,
                        "card_index": item.card_index,
                        "ev": round(item.expected_value, 2),
                        "future": round(item.future_hu_value, 2),
                        "risk": round(item.opponent_risk, 2),
                    }
                    for item in values[:5]
                ],
            )
            if snapshot.screen_type == ScreenType.HU and decision.action != "hu":
                LOGGER.info("当前虽可胡，但EV模型选择继续做牌：%s", decision.reason)
            return decision
        raise RuntimeError(f"当前状态不可决策：{snapshot.screen_type.value}")

    async def _execute(
        self,
        frame: CapturedFrame,
        snapshot: GameSnapshot,
        decision: GameDecision,
    ) -> str:
        if decision.action == "swap":
            assert snapshot.swap_button is not None
            desired = set(decision.card_indexes)
            for index, card in enumerate(snapshot.hand):
                if card.selected and index not in desired:
                    await self._click_card(frame, card)
                    await asyncio.sleep(0.12)
            for index in decision.card_indexes:
                card = snapshot.hand[index]
                if not card.selected:
                    await self._click_card(frame, card)
                    await asyncio.sleep(0.12)
            await self._click(frame, snapshot.swap_button)
            return decision.reason + "；已点击换牌。"

        if decision.action == "draw_visible":
            if decision.visible_index is None:
                raise RuntimeError("draw_visible 缺少 visible_index")
            card = snapshot.visible_cards[decision.visible_index]
            await self._click_card(frame, card)
            self.tracker.record_own_take_visible(card)
            return decision.reason + "；已取亮牌。"

        if decision.action == "draw_deck":
            if snapshot.deck_click is None:
                raise RuntimeError("未识别到牌堆点击位置")
            await self._click(frame, snapshot.deck_click)
            self.tracker.record_own_draw_deck()
            return decision.reason + "；已摸牌堆。"

        if decision.action == "hu":
            if snapshot.hu_button is None:
                raise RuntimeError(
                    f"规则引擎判断已胡（{decision.hu.pattern if decision.hu else '未知'}），"
                    "但视觉没有可靠识别到胡牌按钮；为避免把胡牌打掉，本轮不操作。"
                )
            await self._click(frame, snapshot.hu_button)
            return decision.reason + "；已点击胡牌。"

        if decision.action == "discard":
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
                await asyncio.sleep(0.12)
            await self._click(frame, snapshot.discard_button)
            self.tracker.record_own_discard(desired)
            return decision.reason + "；已出牌。"

        raise RuntimeError(f"未知游戏动作：{decision.action}")

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
            inferences = self.ev.opponent_model.infer_all(
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
        self._recent_opponent_events = {
            signature: seen_at
            for signature, seen_at in self._recent_opponent_events.items()
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
            if signature in self._recent_opponent_events:
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

            self._recent_opponent_events[signature] = now
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
            self._recent_opponent_events.clear()
            self.opponent_event_count = 0
            LOGGER.info("建立新一局记牌状态 gold=%s deck=%s", new_gold, snapshot.deck_count)
            return
        self.tracker.update_own_hand(snapshot.hand)
        self.tracker.set_deck_count(snapshot.deck_count)

    async def _click_card(self, frame: CapturedFrame, card: Card) -> None:
        if card.x is None or card.y is None:
            raise RuntimeError("牌缺少点击坐标")
        await self._click(frame, Point(card.x, card.y))

    async def _click(self, frame: CapturedFrame, point: Point) -> None:
        if not (0 <= point.x < frame.width and 0 <= point.y < frame.height):
            raise RuntimeError(
                f"视觉坐标超出游戏截图：({point.x},{point.y}) / {frame.width}x{frame.height}"
            )
        await asyncio.to_thread(assert_game_foreground, self.window_title_keyword)
        screen_x = frame.origin_x + point.x
        screen_y = frame.origin_y + point.y

        import pyautogui

        self.session.set_native_call(True)
        try:
            await asyncio.to_thread(pyautogui.click, screen_x, screen_y)
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
            ScreenType.SWAP: 7,
            ScreenType.DRAW: 7,
            ScreenType.DISCARD: 8,
            ScreenType.HU: 8,
        }.get(snapshot.screen_type)
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
        if snapshot.screen_type == ScreenType.DRAW and snapshot.deck_click is None and not snapshot.visible_cards:
            return "摸牌阶段既未识别到牌堆，也没有亮牌，暂不操作。"
        if snapshot.screen_type == ScreenType.DISCARD and snapshot.discard_button is None:
            return "出牌阶段未识别到“出牌”按钮，暂不操作。"
        return None

    @staticmethod
    def _snapshot_summary(snapshot: GameSnapshot) -> str:
        hand = " ".join(card.label + ("*" if card.selected else "") for card in snapshot.hand) or "无"
        visible = " ".join(card.label for card in snapshot.visible_cards) or "无"
        gold = "/".join(snapshot.gold_ranks) or "未知"
        deck = "未知" if snapshot.deck_count is None else str(snapshot.deck_count)
        return (
            f"状态：{snapshot.screen_type.value}\n"
            f"金牌：{gold}\n"
            f"手牌({len(snapshot.hand)})：{hand}\n"
            f"亮牌：{visible}\n"
            f"牌堆：{deck}"
        )

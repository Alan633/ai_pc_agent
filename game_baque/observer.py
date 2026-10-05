from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from PIL import Image

from game_baque.capture import grab_game_window
from game_baque.event_vision import OpponentEventVision, TableObservation
from game_baque.motion import MotionEpisode, MotionEpisodeDetector

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ObserverStatus:
    running: bool
    directory: str | None
    sampled_frames: int
    episodes_detected: int
    global_effects_skipped: int
    classified: int
    applied: int
    low_confidence: int
    queue_size: int
    paused_reason: str | None
    recent: tuple[str, ...]

    def to_text(self) -> str:
        lines = [
            f"对手观察器：{'RUNNING' if self.running else 'IDLE'}",
            f"采样帧：{self.sampled_frames}",
            f"动作片段：{self.episodes_detected}",
            f"全屏特效跳过：{self.global_effects_skipped}",
            f"已识别：{self.classified}",
            f"已写入记牌器：{self.applied}",
            f"低置信/未采用：{self.low_confidence}",
            f"待识别队列：{self.queue_size}",
        ]
        if self.directory:
            lines.append(f"目录：{self.directory}")
        if self.paused_reason:
            lines.append(f"暂停原因：{self.paused_reason}")
        if self.recent:
            lines.append("最近识别：")
            lines.extend(f"- {item}" for item in self.recent[-6:])
        return "\n".join(lines)


class OpponentObserver:
    """本地高频观察 + 三关键帧视觉识别。

    捕获线程不会等待远程模型；动作片段进入 asyncio.Queue 后由独立 worker 串行识别，
    因此视觉 API 的网络延迟不会导致下一名玩家的画面直接漏掉。
    """

    def __init__(
        self,
        *,
        vision: OpponentEventVision,
        output_root: Path,
        on_observation: Callable[[TableObservation], bool],
        window_title_keyword: str = "欢乐斗地主",
        fps: float = 6.0,
        min_confidence: float = 0.82,
        max_queue: int = 10,
    ) -> None:
        self.vision = vision
        self.output_root = output_root
        self.on_observation = on_observation
        self.window_title_keyword = window_title_keyword
        self.fps = min(max(float(fps), 2.0), 12.0)
        self.min_confidence = min(max(float(min_confidence), 0.5), 0.99)
        self.max_queue = max(3, int(max_queue))

        self._capture_task: asyncio.Task[None] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._queue: asyncio.Queue[tuple[MotionEpisode, Path, Path, Path]] | None = None
        self._classification_enabled: asyncio.Event | None = None
        self._active_classification: asyncio.Task[TableObservation] | None = None
        self._generation = 0
        self._detector: MotionEpisodeDetector | None = None
        self._directory: Path | None = None
        self._sampled_frames = 0
        self._episodes_detected = 0
        self._global_effects_skipped = 0
        self._classified = 0
        self._applied = 0
        self._low_confidence = 0
        self._paused_reason: str | None = None
        self._recent: deque[str] = deque(maxlen=12)

    @property
    def running(self) -> bool:
        return self._capture_task is not None and not self._capture_task.done()

    async def start(self, *, classification_paused: bool = False) -> ObserverStatus:
        if self.running:
            return self.status()

        _, first = await asyncio.to_thread(
            grab_game_window,
            self.window_title_keyword,
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = self.output_root / f"live_watch_{stamp}"
        suffix = 1
        while directory.exists():
            directory = self.output_root / f"live_watch_{stamp}_{suffix:02d}"
            suffix += 1
        (directory / "episodes").mkdir(parents=True, exist_ok=False)

        self._directory = directory
        self._sampled_frames = 0
        self._episodes_detected = 0
        self._global_effects_skipped = 0
        self._classified = 0
        self._applied = 0
        self._low_confidence = 0
        self._paused_reason = None
        self._recent.clear()
        self._stop_event = asyncio.Event()
        self._queue = asyncio.Queue(maxsize=self.max_queue)
        self._classification_enabled = asyncio.Event()
        if classification_paused:
            self._generation = await self.vision.enter_own_phase()
        else:
            self._classification_enabled.set()
            self._generation = await self.vision.enter_opponent_phase()

        detector = MotionEpisodeDetector(fps=self.fps)
        self._detector = detector
        self._capture_task = asyncio.create_task(
            self._capture_loop(first, detector),
            name="baque-opponent-capture",
        )
        self._worker_task = asyncio.create_task(
            self._classify_loop(),
            name="baque-opponent-classifier",
        )
        LOGGER.info("对手观察器启动 dir=%s fps=%.1f", directory, self.fps)
        return self.status()

    async def pause_classification(self) -> None:
        """自己的回合立即让出本地模型，并使所有旧结果失效。"""
        enabled = self._classification_enabled
        if enabled is None or not enabled.is_set():
            return
        enabled.clear()
        self._generation = await self.vision.enter_own_phase()
        if self._detector is not None:
            self._detector.reset()
        self._drain_queue()
        task = self._active_classification
        if task is not None and not task.done():
            task.cancel()
        self._recent.append("自己的回合：已暂停对手识别并清空旧事件")

    async def resume_classification(self) -> None:
        """进入对手回合后恢复分类；重复通知不会重置动作分段。"""
        enabled = self._classification_enabled
        if enabled is None or enabled.is_set():
            return
        self._drain_queue()
        self._generation = await self.vision.enter_opponent_phase()
        enabled.set()
        self._recent.append("对手回合：已恢复桌面动作识别")

    def _drain_queue(self) -> None:
        if self._queue is None:
            return
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()

    async def stop(self) -> ObserverStatus:
        if self._capture_task is None and self._worker_task is None:
            return self.status()
        if self._stop_event is not None:
            self._stop_event.set()
        if self._classification_enabled is not None:
            self._classification_enabled.set()
        task = self._active_classification
        if task is not None and not task.done():
            task.cancel()

        # 先让捕获循环结束并 flush 最后一个动作片段，再等待识别队列消费完。
        capture_task = self._capture_task
        if capture_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(capture_task), timeout=5.0)
            except asyncio.TimeoutError:
                capture_task.cancel()
                try:
                    await capture_task
                except asyncio.CancelledError:
                    pass

        if self._queue is not None:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=12.0)
            except asyncio.TimeoutError:
                LOGGER.warning("停止对手观察时仍有未完成的视觉识别，将取消剩余 worker")

        worker_task = self._worker_task
        if worker_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(worker_task), timeout=3.0)
            except asyncio.TimeoutError:
                worker_task.cancel()
                try:
                    await worker_task
                except asyncio.CancelledError:
                    pass

        self._capture_task = None
        self._worker_task = None
        self._active_classification = None
        self._classification_enabled = None
        self._detector = None
        self._write_summary()
        LOGGER.info("对手观察器停止")
        return self.status()

    def status(self) -> ObserverStatus:
        return ObserverStatus(
            running=self.running,
            directory=str(self._directory) if self._directory else None,
            sampled_frames=self._sampled_frames,
            episodes_detected=self._episodes_detected,
            global_effects_skipped=self._global_effects_skipped,
            classified=self._classified,
            applied=self._applied,
            low_confidence=self._low_confidence,
            queue_size=self._queue.qsize() if self._queue is not None else 0,
            paused_reason=self._paused_reason,
            recent=tuple(self._recent),
        )

    async def _capture_loop(
        self,
        first: Image.Image,
        detector: MotionEpisodeDetector,
    ) -> None:
        assert self._stop_event is not None
        previous = first.copy()
        period = 1.0 / self.fps
        next_tick = time.monotonic()

        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_tick:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=next_tick - now,
                        )
                        break
                    except asyncio.TimeoutError:
                        pass
                next_tick = max(next_tick + period, time.monotonic())

                try:
                    _, current = await asyncio.to_thread(
                        grab_game_window,
                        self.window_title_keyword,
                    )
                    self._paused_reason = None
                except Exception as exc:
                    self._paused_reason = str(exc)
                    await asyncio.sleep(min(0.5, period * 2.0))
                    continue

                self._sampled_frames += 1
                if (
                    self._classification_enabled is not None
                    and not self._classification_enabled.is_set()
                ):
                    previous = current
                    continue
                episode = detector.push(
                    previous,
                    current,
                    sequence=self._sampled_frames,
                    timestamp=datetime.now(UTC).isoformat(),
                )
                if episode is not None:
                    await self._enqueue_episode(episode)
                previous = current

            pending = detector.flush()
            if pending is not None:
                await self._enqueue_episode(pending)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("对手观察捕获循环异常")

    async def _enqueue_episode(self, episode: MotionEpisode) -> None:
        assert self._directory is not None
        assert self._queue is not None
        self._episodes_detected += 1

        folder = self._directory / "episodes"
        prefix = folder / f"episode_{episode.episode_id:04d}"
        before = prefix.with_name(prefix.name + "_before.jpg")
        peak = prefix.with_name(prefix.name + "_peak.jpg")
        after = prefix.with_name(prefix.name + "_after.jpg")
        await asyncio.gather(
            asyncio.to_thread(self._save_jpeg, episode.before, before),
            asyncio.to_thread(self._save_jpeg, episode.peak, peak),
            asyncio.to_thread(self._save_jpeg, episode.after, after),
        )

        base_record = {
            "episode_id": episode.episode_id,
            "start_sequence": episode.start_sequence,
            "end_sequence": episode.end_sequence,
            "start_timestamp": episode.start_timestamp,
            "end_timestamp": episode.end_timestamp,
            "duration_frames": episode.duration_frames,
            "peak_scores": {
                key: round(value, 6) for key, value in episode.peak_scores.items()
            },
            "overall_peak": round(episode.overall_peak, 6),
            "actor_hint": episode.actor_hint,
            "global_effect": episode.global_effect,
            "before": before.relative_to(self._directory).as_posix(),
            "peak": peak.relative_to(self._directory).as_posix(),
            "after": after.relative_to(self._directory).as_posix(),
        }
        self._append_jsonl(self._directory / "episodes.jsonl", base_record)

        if episode.global_effect:
            self._global_effects_skipped += 1
            self._recent.append(
                f"episode {episode.episode_id}: 跳过全屏特效"
            )
            return

        if (
            self._classification_enabled is not None
            and not self._classification_enabled.is_set()
        ):
            self._recent.append(
                f"episode {episode.episode_id}: 自己回合，仅保留关键帧"
            )
            return

        if self._queue.full():
            self._low_confidence += 1
            self._recent.append(
                f"episode {episode.episode_id}: 识别队列已满，保留图片但暂不调用模型"
            )
            return
        await self._queue.put((episode, before, peak, after))

    async def _classify_loop(self) -> None:
        assert self._stop_event is not None
        assert self._queue is not None
        assert self._classification_enabled is not None

        while True:
            capture_done = self._capture_task is None or self._capture_task.done()
            if self._stop_event.is_set() and capture_done and self._queue.empty():
                break
            await self._classification_enabled.wait()
            try:
                episode, before, peak, after = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.5,
                )
            except asyncio.TimeoutError:
                continue

            try:
                generation = self._generation
                self._active_classification = asyncio.create_task(
                    self.vision.classify(
                        before,
                        peak,
                        after,
                        actor_hint=episode.actor_hint,
                        peak_scores=episode.peak_scores,
                        generation=generation,
                    ),
                    name=f"baque-opponent-episode-{episode.episode_id}",
                )
                observation = await self._active_classification
                if (
                    generation != self._generation
                    or not self._classification_enabled.is_set()
                ):
                    self._recent.append(
                        f"episode {episode.episode_id}: 跨回合旧结果已丢弃"
                    )
                    continue
                self._classified += 1
                applied = False
                error = ""
                if observation.confidence >= self.min_confidence:
                    try:
                        applied = bool(self.on_observation(observation))
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        LOGGER.exception("对手事件写入记牌器失败")
                else:
                    self._low_confidence += 1

                if applied:
                    self._applied += 1
                summary = observation.summary()
                self._recent.append(summary + (" [APPLIED]" if applied else ""))
                self._write_recognition(
                    episode,
                    observation,
                    applied=applied,
                    error=error,
                )
            except asyncio.CancelledError:
                if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                    raise
                self._recent.append(
                    f"episode {episode.episode_id}: 自己回合抢占，识别已取消"
                )
            except Exception as exc:
                self._low_confidence += 1
                self._recent.append(
                    f"episode {episode.episode_id}: 识别失败 {type(exc).__name__}"
                )
                LOGGER.exception("对手动作视觉识别失败")
            finally:
                self._active_classification = None
                self._queue.task_done()

    def _write_recognition(
        self,
        episode: MotionEpisode,
        observation: TableObservation,
        *,
        applied: bool,
        error: str,
    ) -> None:
        if self._directory is None:
            return
        payload = {
            "episode_id": episode.episode_id,
            "actor_hint": episode.actor_hint,
            "observation": {
                "actor": observation.actor,
                "event_type": observation.event_type,
                "draw_source": observation.draw_source,
                "drawn_card": asdict(observation.drawn_card),
                "discard_card": asdict(observation.discard_card),
                "opportunity_cards": [
                    asdict(card) for card in observation.opportunity_cards
                ],
                "opportunity_chosen_index": observation.opportunity_chosen_index,
                "deck_before": observation.deck_before,
                "deck_after": observation.deck_after,
                "confidence": observation.confidence,
                "evidence": observation.evidence,
            },
            "applied": applied,
            "error": error,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        self._append_jsonl(self._directory / "recognized_events.jsonl", payload)

    def _write_summary(self) -> None:
        if self._directory is None:
            return
        payload = {
            **asdict(self.status()),
            "finished_at": datetime.now(UTC).isoformat(),
        }
        try:
            (self._directory / "observer_summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            LOGGER.exception("写入 opponent observer summary 失败")

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _save_jpeg(image: Image.Image, path: Path) -> None:
        image.convert("RGB").save(
            path,
            "JPEG",
            quality=90,
            optimize=True,
            progressive=False,
        )

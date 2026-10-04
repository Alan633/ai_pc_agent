from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat

from game_baque.capture import GameWindow, grab_game_window

LOGGER = logging.getLogger(__name__)


# 归一化 ROI，坐标为 left, top, right, bottom。
# 第一版故意给得稍宽，后续会用真实录像把区域收紧。
DEFAULT_ROIS: dict[str, tuple[float, float, float, float]] = {
    "left_player": (0.00, 0.18, 0.31, 0.69),
    "top_player": (0.28, 0.00, 0.72, 0.34),
    "right_player": (0.69, 0.18, 1.00, 0.69),
    "center_table": (0.22, 0.19, 0.78, 0.66),
    "own_hand": (0.19, 0.68, 0.81, 1.00),
}


@dataclass(frozen=True, slots=True)
class RecorderStatus:
    running: bool
    recording_dir: str | None
    started_at: str | None
    elapsed_seconds: float
    sampled_frames: int
    archived_frames: int
    event_candidates: int
    last_change_score: float
    last_changed_regions: tuple[str, ...]
    paused_reason: str | None

    def to_text(self) -> str:
        if not self.recording_dir:
            return "采集器：未启动"
        state = "录制中" if self.running else "已停止"
        lines = [
            f"采集器：{state}",
            f"目录：{self.recording_dir}",
            f"已运行：{self.elapsed_seconds:.1f}s",
            f"采样帧：{self.sampled_frames}",
            f"归档帧：{self.archived_frames}",
            f"事件候选：{self.event_candidates}",
            f"最近变化：{self.last_change_score:.4f}",
        ]
        if self.last_changed_regions:
            lines.append("变化区域：" + "、".join(self.last_changed_regions))
        if self.paused_reason:
            lines.append("暂停原因：" + self.paused_reason)
        return "\n".join(lines)


def normalized_crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop(
        (
            max(0, min(width, round(left * width))),
            max(0, min(height, round(top * height))),
            max(0, min(width, round(right * width))),
            max(0, min(height, round(bottom * height))),
        )
    )


def image_change_score(
    previous: Image.Image,
    current: Image.Image,
    *,
    sample_size: tuple[int, int] = (192, 108),
) -> float:
    """返回 0~1 的平均像素变化比例，用于轻量级事件门控。"""
    if previous.size != current.size:
        return 1.0
    a = previous.convert("L").resize(sample_size)
    b = current.convert("L").resize(sample_size)
    diff = ImageChops.difference(a, b)
    mean = ImageStat.Stat(diff).mean[0]
    return float(mean) / 255.0


class GameRecorder:
    """点字成金真实牌局采集器。

    - 高频截图只在本地进行，不调用 OpenAI。
    - 以较低 FPS 保存连续归档帧。
    - ROI 变化超过阈值时额外保存前后关键帧和事件 JSONL。
    """

    def __init__(
        self,
        *,
        recordings_dir: Path,
        window_title_keyword: str = "欢乐斗地主",
        fps: float = 6.0,
        archive_fps: float = 2.0,
        change_threshold: float = 0.028,
        jpeg_quality: int = 78,
        event_jpeg_quality: int = 92,
        rois: dict[str, tuple[float, float, float, float]] | None = None,
    ) -> None:
        self.recordings_dir = recordings_dir
        self.window_title_keyword = window_title_keyword
        self.fps = min(max(float(fps), 1.0), 12.0)
        self.archive_fps = min(max(float(archive_fps), 0.2), self.fps)
        self.change_threshold = min(max(float(change_threshold), 0.001), 0.30)
        self.jpeg_quality = min(max(int(jpeg_quality), 40), 95)
        self.event_jpeg_quality = min(max(int(event_jpeg_quality), 60), 98)
        self.rois = dict(rois or DEFAULT_ROIS)

        self._task: asyncio.Task[None] | None = None
        self._stop_event: asyncio.Event | None = None
        self._recording_dir: Path | None = None
        self._started_monotonic: float | None = None
        self._started_at: datetime | None = None
        self._sampled_frames = 0
        self._archived_frames = 0
        self._event_candidates = 0
        self._last_change_score = 0.0
        self._last_changed_regions: tuple[str, ...] = ()
        self._paused_reason: str | None = None
        self._last_window: GameWindow | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> RecorderStatus:
        if self.running:
            return self.status()

        # 先抓一帧，确保用户真的把游戏放在前台。
        window, first_image = await asyncio.to_thread(
            grab_game_window,
            self.window_title_keyword,
        )

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_dir = self.recordings_dir / stamp
        suffix = 1
        while recording_dir.exists():
            recording_dir = self.recordings_dir / f"{stamp}_{suffix:02d}"
            suffix += 1
        (recording_dir / "frames").mkdir(parents=True, exist_ok=False)
        (recording_dir / "events").mkdir(parents=True, exist_ok=False)

        self._recording_dir = recording_dir
        self._started_at = datetime.now(UTC)
        self._started_monotonic = time.monotonic()
        self._sampled_frames = 0
        self._archived_frames = 0
        self._event_candidates = 0
        self._last_change_score = 0.0
        self._last_changed_regions = ()
        self._paused_reason = None
        self._last_window = window
        self._stop_event = asyncio.Event()

        meta = {
            "version": 1,
            "started_at": self._started_at.isoformat(),
            "window_title_keyword": self.window_title_keyword,
            "window": asdict(window),
            "fps": self.fps,
            "archive_fps": self.archive_fps,
            "change_threshold": self.change_threshold,
            "jpeg_quality": self.jpeg_quality,
            "event_jpeg_quality": self.event_jpeg_quality,
            "rois": self.rois,
            "notes": "本地截图采集；不调用视觉模型。events 仅为图像变化候选，尚未自动标注牌局动作。",
        }
        (recording_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 第 0 帧始终保存，后续差分有可靠基线。
        self._save_jpeg(
            first_image,
            recording_dir / "frames" / "frame_000000.jpg",
            self.jpeg_quality,
        )
        self._archived_frames = 1

        self._task = asyncio.create_task(
            self._run(first_image),
            name="baque-game-recorder",
        )
        LOGGER.info("点字成金采集开始 dir=%s fps=%.1f", recording_dir, self.fps)
        return self.status()

    async def stop(self) -> RecorderStatus:
        task = self._task
        if task is None:
            return self.status()
        if self._stop_event is not None:
            self._stop_event.set()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
        except asyncio.TimeoutError:
            LOGGER.warning("点字成金采集器停止等待超时，取消任务")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        finally:
            self._task = None
            self._write_summary()
        LOGGER.info("点字成金采集结束 dir=%s", self._recording_dir)
        return self.status()

    def status(self) -> RecorderStatus:
        elapsed = (
            time.monotonic() - self._started_monotonic
            if self._started_monotonic is not None
            else 0.0
        )
        return RecorderStatus(
            running=self.running,
            recording_dir=str(self._recording_dir) if self._recording_dir else None,
            started_at=self._started_at.isoformat() if self._started_at else None,
            elapsed_seconds=max(0.0, elapsed),
            sampled_frames=self._sampled_frames,
            archived_frames=self._archived_frames,
            event_candidates=self._event_candidates,
            last_change_score=self._last_change_score,
            last_changed_regions=self._last_changed_regions,
            paused_reason=self._paused_reason,
        )

    async def _run(self, first_image: Image.Image) -> None:
        assert self._recording_dir is not None
        assert self._stop_event is not None

        previous = first_image.copy()
        period = 1.0 / self.fps
        archive_every = max(1, round(self.fps / self.archive_fps))
        next_tick = time.monotonic()
        last_event_time = 0.0
        frame_log = self._recording_dir / "frames.jsonl"
        event_log = self._recording_dir / "events.jsonl"

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
                    window, current = await asyncio.to_thread(
                        grab_game_window,
                        self.window_title_keyword,
                    )
                    self._paused_reason = None
                    self._last_window = window
                except Exception as exc:
                    # 用户短暂切窗时不要结束整局录制；暂停采样，回来后继续。
                    self._paused_reason = str(exc)
                    await asyncio.sleep(min(0.5, period * 2.0))
                    continue

                self._sampled_frames += 1
                sequence = self._sampled_frames
                changed_regions, region_scores = self._region_changes(previous, current)
                overall_score = image_change_score(previous, current)
                self._last_change_score = overall_score
                self._last_changed_regions = tuple(changed_regions)
                timestamp = datetime.now(UTC).isoformat()

                archived_path: str | None = None
                if sequence % archive_every == 0:
                    archive_file = self._recording_dir / "frames" / f"frame_{sequence:06d}.jpg"
                    await asyncio.to_thread(
                        self._save_jpeg,
                        current,
                        archive_file,
                        self.jpeg_quality,
                    )
                    self._archived_frames += 1
                    archived_path = archive_file.relative_to(self._recording_dir).as_posix()

                frame_record: dict[str, Any] = {
                    "sequence": sequence,
                    "timestamp": timestamp,
                    "overall_change": round(overall_score, 6),
                    "region_change": {
                        key: round(value, 6) for key, value in region_scores.items()
                    },
                    "changed_regions": changed_regions,
                    "archive": archived_path,
                    "window": {
                        "x": window.origin_x,
                        "y": window.origin_y,
                        "width": window.width,
                        "height": window.height,
                    },
                }
                self._append_jsonl(frame_log, frame_record)

                event_score = max(region_scores.values(), default=overall_score)
                event_now = time.monotonic()
                if (
                    changed_regions
                    and event_score >= self.change_threshold
                    and event_now - last_event_time >= 0.28
                ):
                    self._event_candidates += 1
                    event_id = self._event_candidates
                    before_path = self._recording_dir / "events" / f"event_{event_id:04d}_before.jpg"
                    after_path = self._recording_dir / "events" / f"event_{event_id:04d}_after.jpg"
                    await asyncio.gather(
                        asyncio.to_thread(
                            self._save_jpeg,
                            previous,
                            before_path,
                            self.event_jpeg_quality,
                        ),
                        asyncio.to_thread(
                            self._save_jpeg,
                            current,
                            after_path,
                            self.event_jpeg_quality,
                        ),
                    )
                    self._append_jsonl(
                        event_log,
                        {
                            "event_id": event_id,
                            "sequence": sequence,
                            "timestamp": timestamp,
                            "overall_change": round(overall_score, 6),
                            "regions": changed_regions,
                            "region_change": {
                                key: round(value, 6)
                                for key, value in region_scores.items()
                                if key in changed_regions
                            },
                            "before": before_path.relative_to(self._recording_dir).as_posix(),
                            "after": after_path.relative_to(self._recording_dir).as_posix(),
                            "label": None,
                        },
                    )
                    last_event_time = event_now

                previous = current
        except asyncio.CancelledError:
            raise
        except Exception:
            LOGGER.exception("点字成金采集循环异常")
        finally:
            self._write_summary()

    def _region_changes(
        self,
        previous: Image.Image,
        current: Image.Image,
    ) -> tuple[list[str], dict[str, float]]:
        scores: dict[str, float] = {}
        changed: list[str] = []
        for name, box in self.rois.items():
            prev_crop = normalized_crop(previous, box)
            curr_crop = normalized_crop(current, box)
            score = image_change_score(prev_crop, curr_crop, sample_size=(128, 96))
            scores[name] = score
            if score >= self.change_threshold:
                changed.append(name)
        changed.sort(key=lambda name: scores[name], reverse=True)
        return changed, scores

    def _write_summary(self) -> None:
        if self._recording_dir is None:
            return
        status = self.status()
        payload = {
            **asdict(status),
            "finished_at": datetime.now(UTC).isoformat(),
            "window": asdict(self._last_window) if self._last_window else None,
        }
        try:
            (self._recording_dir / "summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            LOGGER.exception("写入采集 summary.json 失败")

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(
            path,
            "JPEG",
            quality=quality,
            optimize=True,
            progressive=False,
        )

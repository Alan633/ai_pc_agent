from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from PIL import Image

from game_baque.recorder_utils import image_change_score, normalized_crop


# 2026-10-04 首轮真实牌局（1916x1076）校准后的窄 ROI。
# 与旧版“玩家大区域”相比，不再把中央特效大面积包含到三家玩家区域中。
ACTION_ROIS: dict[str, tuple[float, float, float, float]] = {
    "left_hand": (0.075, 0.25, 0.155, 0.53),
    "top_hand": (0.33, 0.035, 0.46, 0.16),
    "right_hand": (0.835, 0.25, 0.915, 0.53),
    "center_cards": (0.33, 0.25, 0.70, 0.55),
    "own_hand": (0.25, 0.70, 0.76, 0.995),
}

SEAT_REGION = {
    "left": "left_hand",
    "top": "top_hand",
    "right": "right_hand",
    "self": "own_hand",
}


@dataclass(frozen=True, slots=True)
class MotionEpisode:
    episode_id: int
    start_sequence: int
    end_sequence: int
    start_timestamp: str
    end_timestamp: str
    before: Image.Image
    peak: Image.Image
    after: Image.Image
    peak_scores: dict[str, float]
    overall_peak: float
    actor_hint: str
    global_effect: bool

    @property
    def duration_frames(self) -> int:
        return max(1, self.end_sequence - self.start_sequence + 1)


class MotionEpisodeDetector:
    """把一串 6 FPS 左右的瞬时差分合并成“一个动作片段”。

    首轮样本 250.7 秒产生了 145 个 raw event，但按约 1.2 秒动作间隔只形成
    约 39 个动作簇。实时检测器因此不应该对每一个动画帧调用视觉模型，而应该先
    合并为 before / peak / after 三张关键帧。
    """

    def __init__(
        self,
        *,
        trigger_threshold: float = 0.018,
        stable_threshold: float = 0.006,
        stable_frames: int = 3,
        max_duration_seconds: float = 5.5,
        fps: float = 6.0,
        rois: Mapping[str, tuple[float, float, float, float]] | None = None,
    ) -> None:
        self.trigger_threshold = max(0.003, float(trigger_threshold))
        self.stable_threshold = max(0.001, min(float(stable_threshold), self.trigger_threshold))
        self.stable_frames = max(2, int(stable_frames))
        self.max_duration_frames = max(
            self.stable_frames + 1,
            int(max_duration_seconds * max(1.0, float(fps))),
        )
        self.rois = dict(rois or ACTION_ROIS)

        self._next_id = 1
        self._active = False
        self._before: Image.Image | None = None
        self._peak: Image.Image | None = None
        self._after: Image.Image | None = None
        self._start_sequence = 0
        self._start_timestamp = ""
        self._end_sequence = 0
        self._end_timestamp = ""
        self._stable_count = 0
        self._peak_scores: dict[str, float] = {}
        self._overall_peak = 0.0
        self._peak_metric = 0.0

    def region_scores(
        self,
        previous: Image.Image,
        current: Image.Image,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name, box in self.rois.items():
            scores[name] = image_change_score(
                normalized_crop(previous, box),
                normalized_crop(current, box),
                sample_size=(96, 96),
            )
        return scores

    def push(
        self,
        previous: Image.Image,
        current: Image.Image,
        *,
        sequence: int,
        timestamp: str,
    ) -> MotionEpisode | None:
        scores = self.region_scores(previous, current)
        overall = image_change_score(previous, current, sample_size=(192, 108))
        metric = max(scores.values(), default=overall)

        if not self._active:
            if metric < self.trigger_threshold:
                return None
            self._active = True
            self._before = previous.copy()
            self._peak = current.copy()
            self._after = current.copy()
            self._start_sequence = sequence
            self._end_sequence = sequence
            self._start_timestamp = timestamp
            self._end_timestamp = timestamp
            self._stable_count = 0
            self._peak_scores = dict(scores)
            self._overall_peak = overall
            self._peak_metric = metric
            return None

        self._after = current.copy()
        self._end_sequence = sequence
        self._end_timestamp = timestamp
        for name, value in scores.items():
            self._peak_scores[name] = max(self._peak_scores.get(name, 0.0), value)
        self._overall_peak = max(self._overall_peak, overall)
        if metric > self._peak_metric:
            self._peak_metric = metric
            self._peak = current.copy()

        if metric <= self.stable_threshold:
            self._stable_count += 1
        else:
            self._stable_count = 0

        too_long = (sequence - self._start_sequence + 1) >= self.max_duration_frames
        if self._stable_count >= self.stable_frames or too_long:
            return self._finish()
        return None

    def flush(self) -> MotionEpisode | None:
        if not self._active:
            return None
        return self._finish()

    def _finish(self) -> MotionEpisode:
        assert self._before is not None
        assert self._peak is not None
        assert self._after is not None

        actor_hint, global_effect = infer_actor_hint(
            self._peak_scores,
            self._overall_peak,
        )
        episode = MotionEpisode(
            episode_id=self._next_id,
            start_sequence=self._start_sequence,
            end_sequence=self._end_sequence,
            start_timestamp=self._start_timestamp,
            end_timestamp=self._end_timestamp,
            before=self._before,
            peak=self._peak,
            after=self._after,
            peak_scores=dict(self._peak_scores),
            overall_peak=self._overall_peak,
            actor_hint=actor_hint,
            global_effect=global_effect,
        )
        self._next_id += 1
        self._reset()
        return episode

    def _reset(self) -> None:
        self._active = False
        self._before = None
        self._peak = None
        self._after = None
        self._start_sequence = 0
        self._start_timestamp = ""
        self._end_sequence = 0
        self._end_timestamp = ""
        self._stable_count = 0
        self._peak_scores = {}
        self._overall_peak = 0.0
        self._peak_metric = 0.0


def infer_actor_hint(
    peak_scores: Mapping[str, float],
    overall_peak: float,
) -> tuple[str, bool]:
    seat_scores = {
        seat: float(peak_scores.get(region, 0.0))
        for seat, region in SEAT_REGION.items()
    }
    ranked = sorted(seat_scores.items(), key=lambda item: item[1], reverse=True)
    high_regions = sum(value >= 0.08 for value in peak_scores.values())

    # “硬气”“比翼为邻”“胡牌提示”等大特效会同时扫过多块区域；
    # 首轮样本中这类帧的 overall/多 ROI 峰值显著高于普通单家动作。
    global_effect = overall_peak >= 0.10 and high_regions >= 3
    if global_effect or not ranked:
        return "unknown", global_effect

    best_seat, best = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if best < 0.025:
        return "unknown", False
    if second > 0 and best < second * 1.18 and best - second < 0.015:
        return "unknown", False
    return best_seat, False

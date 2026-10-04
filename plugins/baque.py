from __future__ import annotations

from dataclasses import dataclass

from agent.session import AgentSession
from config import Settings
from game_baque.controller import BaqueGameController
from game_baque.event_vision import OpponentEventVision
from game_baque.observer import OpponentObserver
from game_baque.recorder import GameRecorder
from game_baque.vision import BaqueVision


@dataclass(slots=True)
class BaquePlugin:
    """点字成金能力插件。

    这里只负责组装和生命周期管理。通用 Agent 主程序不再直接关心
    game_baque 内部的 vision/tracker/EV/controller 实现。
    """

    game: BaqueGameController | None
    recorder: GameRecorder
    observer: OpponentObserver | None

    @classmethod
    def build(cls, settings: Settings, session: AgentSession) -> "BaquePlugin":
        recorder = GameRecorder(
            recordings_dir=settings.game_recordings_dir,
            window_title_keyword=settings.game_window_title_keyword,
            fps=settings.game_record_fps,
            archive_fps=settings.game_record_archive_fps,
            change_threshold=settings.game_record_diff_threshold,
            jpeg_quality=settings.game_record_jpeg_quality,
        )

        game: BaqueGameController | None = None
        if settings.openai_api_key or settings.game_vision_base_url:
            game = BaqueGameController(
                vision=BaqueVision(
                    settings.openai_api_key,
                    settings.game_vision_model,
                    settings.game_vision_base_url,
                ),
                screenshots_dir=settings.screenshots_dir,
                session=session,
                window_title_keyword=settings.game_window_title_keyword,
                action_interval_seconds=settings.game_action_interval_seconds,
                min_card_confidence=settings.game_min_card_confidence,
                min_event_confidence=settings.game_min_event_confidence,
                auto_next_round=settings.game_auto_next_round,
            )

        observer: OpponentObserver | None = None
        if game is not None:
            observer = OpponentObserver(
                vision=OpponentEventVision(
                    settings.openai_api_key,
                    settings.game_vision_model,
                    base_url=settings.game_vision_base_url,
                ),
                output_root=settings.game_recordings_dir,
                on_observation=game.apply_opponent_observation,
                window_title_keyword=settings.game_window_title_keyword,
                fps=settings.game_record_fps,
                min_confidence=settings.game_min_event_confidence,
            )

        return cls(game=game, recorder=recorder, observer=observer)

    async def shutdown(self) -> None:
        if self.recorder.running:
            await self.recorder.stop()
        if self.observer is not None and self.observer.running:
            await self.observer.stop()
        if self.game is not None:
            self.game.set_external_observer_active(False)

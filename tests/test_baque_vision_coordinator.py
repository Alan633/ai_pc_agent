from __future__ import annotations

import asyncio

import pytest

from game_baque.observer import OpponentObserver
from game_baque.vision_coordinator import (
    LocalVisionCoordinator,
    StaleOpponentRequestError,
)


@pytest.mark.asyncio
async def test_own_phase_cancels_and_rejects_late_opponent_result() -> None:
    coordinator = LocalVisionCoordinator()
    generation = await coordinator.enter_opponent_phase()
    started = asyncio.Event()
    release = asyncio.Event()

    async def cancellation_resistant_request() -> str:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # 模拟底层 HTTP/推理后端吞掉取消并迟到返回。
            return "late"
        return "normal"

    opponent = asyncio.create_task(
        coordinator.run_opponent(generation, cancellation_resistant_request)
    )
    await started.wait()
    await coordinator.enter_own_phase()

    with pytest.raises(StaleOpponentRequestError):
        await opponent


@pytest.mark.asyncio
async def test_own_request_runs_before_queued_opponent_request() -> None:
    coordinator = LocalVisionCoordinator()
    generation = await coordinator.enter_opponent_phase()
    await coordinator.enter_own_phase()
    order: list[str] = []

    stale_opponent = asyncio.create_task(
        coordinator.run_opponent(
            generation,
            lambda: _record(order, "opponent"),
        )
    )
    own = asyncio.create_task(
        coordinator.run_own(lambda: _record(order, "own"))
    )

    assert await own == "own"
    with pytest.raises(StaleOpponentRequestError):
        await stale_opponent
    assert order == ["own"]


async def _record(order: list[str], value: str) -> str:
    order.append(value)
    return value


class _FakeOpponentVision:
    def __init__(self) -> None:
        self.own_entries = 0
        self.opponent_entries = 0

    async def enter_own_phase(self) -> int:
        self.own_entries += 1
        return self.own_entries

    async def enter_opponent_phase(self) -> int:
        self.opponent_entries += 1
        return self.own_entries


class _FakeDetector:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


@pytest.mark.asyncio
async def test_observer_pause_is_idempotent_and_drains_old_episodes(tmp_path) -> None:
    vision = _FakeOpponentVision()
    observer = OpponentObserver(
        vision=vision,  # type: ignore[arg-type]
        output_root=tmp_path,
        on_observation=lambda _observation: True,
    )
    observer._classification_enabled = asyncio.Event()
    observer._classification_enabled.set()
    observer._queue = asyncio.Queue()
    observer._queue.put_nowait((object(), tmp_path, tmp_path, tmp_path))  # type: ignore[arg-type]
    detector = _FakeDetector()
    observer._detector = detector  # type: ignore[assignment]

    await observer.pause_classification()
    await observer.pause_classification()

    assert observer._queue.empty()
    assert not observer._classification_enabled.is_set()
    assert detector.resets == 1
    assert vision.own_entries == 1

    await observer.resume_classification()
    await observer.resume_classification()

    assert observer._classification_enabled.is_set()
    assert vision.opponent_entries == 1

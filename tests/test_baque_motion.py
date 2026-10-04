from PIL import Image, ImageDraw

from game_baque.motion import MotionEpisodeDetector


def test_motion_episode_infers_left_actor() -> None:
    base = Image.new("RGB", (1000, 600), "black")
    changed = base.copy()
    draw = ImageDraw.Draw(changed)
    # left_hand ROI: x 75~155, y 150~318
    draw.rectangle((80, 165, 150, 300), fill="white")

    detector = MotionEpisodeDetector(
        fps=6,
        trigger_threshold=0.01,
        stable_threshold=0.002,
        stable_frames=2,
    )
    assert detector.push(
        base, changed, sequence=1, timestamp="2026-10-04T00:00:00+00:00"
    ) is None
    assert detector.push(
        changed, changed, sequence=2, timestamp="2026-10-04T00:00:00.2+00:00"
    ) is None
    episode = detector.push(
        changed, changed, sequence=3, timestamp="2026-10-04T00:00:00.4+00:00"
    )
    assert episode is not None
    assert episode.actor_hint == "left"
    assert episode.global_effect is False


def test_motion_episode_marks_global_effect() -> None:
    base = Image.new("RGB", (1000, 600), "black")
    changed = Image.new("RGB", (1000, 600), "white")
    detector = MotionEpisodeDetector(
        fps=6,
        trigger_threshold=0.01,
        stable_threshold=0.002,
        stable_frames=2,
    )
    detector.push(base, changed, sequence=1, timestamp="a")
    detector.push(changed, changed, sequence=2, timestamp="b")
    episode = detector.push(changed, changed, sequence=3, timestamp="c")
    assert episode is not None
    assert episode.global_effect is True
    assert episode.actor_hint == "unknown"

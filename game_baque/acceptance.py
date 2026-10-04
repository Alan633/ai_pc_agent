from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class _GameLike(Protocol):
    action_count: int
    opponent_event_count: int
    tracker: Any
    last_message: str

    def quality_metrics(self) -> dict[str, Any]: ...
    def state_text(self) -> str: ...


class _ObserverLike(Protocol):
    def status(self) -> Any: ...


@dataclass(slots=True)
class ManualAccuracy:
    checks: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float | None:
        if self.checks <= 0:
            return None
        return self.correct / self.checks

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "correct": self.correct,
            "incorrect": self.checks - self.correct,
            "accuracy": round(self.accuracy, 4) if self.accuracy is not None else None,
        }


class BaqueAcceptanceTracker:
    """八雀真实牌局验收统计。

    自动指标只能衡量 unknown、低置信、队列丢弃、牌池冲突等“可观测质量”。
    真正的识别准确率必须有人工真值，因此提供三类轻量标记：
    own_state / opponent_event / action。标记会追加到 JSONL，重启后仍可追溯。
    """

    CATEGORIES = ("own_state", "opponent_event", "action")

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.labels_path = self.output_root / "acceptance_labels.jsonl"
        self.report_path = self.output_root / "acceptance_report.json"
        self._lock = threading.Lock()
        self._stats = {name: ManualAccuracy() for name in self.CATEGORIES}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.labels_path.is_file():
            return
        try:
            for line in self.labels_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("event") == "reset":
                    self._stats = {name: ManualAccuracy() for name in self.CATEGORIES}
                    continue
                category = payload.get("category")
                correct = payload.get("correct")
                if category in self._stats and isinstance(correct, bool):
                    stat = self._stats[category]
                    stat.checks += 1
                    stat.correct += int(correct)
        except (OSError, json.JSONDecodeError):
            # 历史统计损坏不应该阻止 Agent 启动；新标记仍继续追加。
            self._stats = {name: ManualAccuracy() for name in self.CATEGORIES}

    def mark(
        self,
        category: str,
        correct: bool,
        *,
        note: str = "",
        context: str = "",
    ) -> dict[str, Any]:
        normalized = category.strip().lower()
        if normalized not in self._stats:
            raise ValueError(
                "category 必须是 own_state、opponent_event 或 action"
            )
        if len(note) > 500:
            note = note[:500]
        if len(context) > 2000:
            context = context[:2000]

        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "category": normalized,
            "correct": bool(correct),
            "note": note,
            "context": context,
        }
        with self._lock:
            stat = self._stats[normalized]
            stat.checks += 1
            stat.correct += int(bool(correct))
            with self.labels_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            snapshot = stat.to_dict()
        return {"category": normalized, **snapshot}

    def reset_session(self) -> dict[str, Any]:
        """开始一轮新的验收统计，不删除历史 JSONL。"""
        with self._lock:
            self._stats = {name: ManualAccuracy() for name in self.CATEGORIES}
            payload = {
                "timestamp": datetime.now(UTC).isoformat(),
                "event": "reset",
            }
            with self.labels_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return {"reset": True, "labels_path": str(self.labels_path)}

    def report(self, game: _GameLike | None, observer: _ObserverLike | None) -> dict[str, Any]:
        with self._lock:
            manual = {name: stat.to_dict() for name, stat in self._stats.items()}

        game_metrics: dict[str, Any] = {}
        tracker_ok = False
        tracker_error: str | None = None
        if game is not None:
            game_metrics = dict(game.quality_metrics())
            try:
                unseen = game.tracker.unseen_counts()
                tracker_ok = all(value >= 0 for value in unseen.values())
            except Exception as exc:
                tracker_error = f"{type(exc).__name__}: {exc}"

        observer_metrics: dict[str, Any] = {}
        if observer is not None:
            status = observer.status()
            observer_metrics = asdict(status)
            classified = int(observer_metrics.get("classified", 0))
            applied = int(observer_metrics.get("applied", 0))
            low_confidence = int(observer_metrics.get("low_confidence", 0))
            episodes = int(observer_metrics.get("episodes_detected", 0))
            global_skipped = int(observer_metrics.get("global_effects_skipped", 0))
            actionable_episodes = max(0, episodes - global_skipped)
            observer_metrics["applied_rate"] = (
                round(applied / classified, 4) if classified else None
            )
            observer_metrics["low_confidence_rate"] = (
                round(low_confidence / actionable_episodes, 4)
                if actionable_episodes
                else None
            )

        checks = {
            "own_state": self._manual_check(manual["own_state"], minimum=20, target=0.98),
            "opponent_event": self._manual_check(
                manual["opponent_event"], minimum=30, target=0.95
            ),
            "action": self._manual_check(manual["action"], minimum=20, target=0.99),
        }

        inspect_count = int(game_metrics.get("inspect_count", 0))
        unknown_count = int(game_metrics.get("unknown_snapshot_count", 0))
        unknown_rate = unknown_count / inspect_count if inspect_count else None
        auto_checks = {
            "tracker_consistency": {
                "status": "PASS" if tracker_ok else "FAIL",
                "detail": tracker_error or "牌池计数未发现冲突",
            },
            "own_unknown_rate": {
                "status": (
                    "INSUFFICIENT_DATA"
                    if inspect_count < 20
                    else ("PASS" if (unknown_rate or 0.0) <= 0.10 else "FAIL")
                ),
                "value": round(unknown_rate, 4) if unknown_rate is not None else None,
                "target_max": 0.10,
                "samples": inspect_count,
            },
        }

        observer_classified = int(observer_metrics.get("classified", 0))
        observer_low_rate = observer_metrics.get("low_confidence_rate")
        auto_checks["observer_stability"] = {
            "status": (
                "INSUFFICIENT_DATA"
                if observer_classified < 30
                else (
                    "PASS"
                    if observer_low_rate is not None and observer_low_rate <= 0.20
                    else "FAIL"
                )
            ),
            "classified": observer_classified,
            "low_confidence_rate": observer_low_rate,
            "target_max": 0.20,
        }

        statuses = [
            item["status"] for item in checks.values()
        ] + [item["status"] for item in auto_checks.values()]
        if "FAIL" in statuses:
            verdict = "FAIL"
        elif "INSUFFICIENT_DATA" in statuses:
            verdict = "INSUFFICIENT_DATA"
        else:
            verdict = "PASS"

        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "verdict": verdict,
            "manual_accuracy": manual,
            "manual_checks": checks,
            "automatic_checks": auto_checks,
            "game": game_metrics,
            "observer": observer_metrics,
            "labels_path": str(self.labels_path),
        }
        try:
            self.report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass
        return report

    @staticmethod
    def _manual_check(
        metric: dict[str, Any],
        *,
        minimum: int,
        target: float,
    ) -> dict[str, Any]:
        checks = int(metric["checks"])
        accuracy = metric["accuracy"]
        if checks < minimum:
            status = "INSUFFICIENT_DATA"
        elif accuracy is not None and accuracy >= target:
            status = "PASS"
        else:
            status = "FAIL"
        return {
            "status": status,
            "checks": checks,
            "minimum_checks": minimum,
            "accuracy": accuracy,
            "target_min": target,
        }

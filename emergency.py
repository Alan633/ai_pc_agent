from __future__ import annotations

import logging
from typing import Any

from agent.session import AgentSession

LOGGER = logging.getLogger(__name__)


class EmergencyHotkey:
    def __init__(self, session: AgentSession) -> None:
        self.session = session
        self.listener: Any = None

    def start(self) -> None:
        try:
            from pynput import keyboard

            self.listener = keyboard.GlobalHotKeys({"<ctrl>+<alt>+<f12>": self._trigger})
            self.listener.start()
        except Exception:
            LOGGER.exception("紧急停止热键启动失败")

    def _trigger(self) -> None:
        LOGGER.warning("触发本地紧急停止 Ctrl+Alt+F12")
        self.session.request_stop_threadsafe()

    def stop(self) -> None:
        if self.listener:
            self.listener.stop()


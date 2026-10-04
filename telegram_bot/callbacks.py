from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from time import monotonic


@dataclass(slots=True)
class PendingConfirmation:
    user_id: int
    future: asyncio.Future[bool]
    expires_at: float


class ConfirmationStore:
    def __init__(self) -> None:
        self._items: dict[str, PendingConfirmation] = {}

    def create(self, user_id: int, timeout: int) -> tuple[str, asyncio.Future[bool]]:
        token = secrets.token_urlsafe(12)
        future = asyncio.get_running_loop().create_future()
        self._items[token] = PendingConfirmation(user_id, future, monotonic() + timeout)
        return token, future

    def resolve(self, token: str, user_id: int, approved: bool) -> bool:
        item = self._items.pop(token, None)
        if item is None or item.user_id != user_id or monotonic() > item.expires_at:
            if item and not item.future.done():
                item.future.set_result(False)
            return False
        if not item.future.done():
            item.future.set_result(approved)
        return True

    def cancel_all(self) -> None:
        for item in self._items.values():
            if not item.future.done():
                item.future.set_result(False)
        self._items.clear()

    def discard(self, token: str) -> None:
        self._items.pop(token, None)


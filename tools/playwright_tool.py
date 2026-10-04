from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse


class PlaywrightController:
    """所有 Playwright sync API 调用固定在同一专用线程，避免 greenlet 跨线程。"""

    def __init__(self, screenshots_dir: Path, *, headless: bool = False) -> None:
        self.screenshots_dir = screenshots_dir
        self.headless = headless
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="playwright")
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._counter = 0
        self._closed = False
        self._lock = threading.Lock()

    def _submit(self, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Playwright 已关闭")
            future = self._executor.submit(function)
        return future.result(timeout=90)

    def _ensure(self) -> None:
        if self._playwright is None:
            from playwright.sync_api import sync_playwright

            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._page = self._browser.new_page()

    def open(self, url: str) -> dict[str, Any]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("只允许 http/https URL")

        def operation() -> dict[str, Any]:
            self._ensure()
            self._page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            return {"url": self._page.url, "title": self._page.title()}

        return self._submit(operation)

    def click(self, selector: str | None = None, text: str | None = None) -> dict[str, Any]:
        if bool(selector) == bool(text):
            raise ValueError("selector 和 text 必须且只能提供一个")

        def operation() -> dict[str, Any]:
            self._ensure()
            locator = self._page.locator(selector) if selector else self._page.get_by_text(text, exact=True)
            locator.first.click(timeout=15_000)
            return {"clicked": selector or text, "url": self._page.url}

        return self._submit(operation)

    def fill(self, selector: str, text: str) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            self._ensure()
            self._page.locator(selector).first.fill(text, timeout=15_000)
            return {"filled": selector, "length": len(text)}

        return self._submit(operation)

    def screenshot(self) -> dict[str, Any]:
        def operation() -> dict[str, Any]:
            self._ensure()
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
            self._counter += 1
            path = self.screenshots_dir / f"playwright_{self._counter:04d}.png"
            self._page.screenshot(path=str(path), full_page=True)
            return {"path": str(path), "url": self._page.url}

        return self._submit(operation)

    def close(self) -> None:
        def operation() -> dict[str, Any]:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
            return {"closed": True}

        try:
            if not self._closed:
                self._submit(operation)
        finally:
            self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=True)

from __future__ import annotations

import os
import subprocess
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


APP_CANDIDATES: dict[str, list[Path]] = {
    "chrome": [
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    ],
    "idea": [
        Path(os.environ.get("LOCALAPPDATA", "")) / "JetBrains/Toolbox/apps/IDEA-U/ch-0/bin/idea64.exe",
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "JetBrains/IntelliJ IDEA/bin/idea64.exe",
    ],
    "notepad": [Path(os.environ.get("WINDIR", "C:/Windows")) / "System32/notepad.exe"],
    "explorer": [Path(os.environ.get("WINDIR", "C:/Windows")) / "explorer.exe"],
}


def _find_jetbrains_idea() -> Path | None:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "JetBrains/Toolbox/apps"
    if base.exists():
        matches = sorted(base.glob("**/bin/idea64.exe"), reverse=True)
        return matches[0] if matches else None
    return None


def launch_app(app_name: str) -> dict[str, Any]:
    normalized = app_name.strip().lower()
    if normalized not in APP_CANDIDATES:
        raise ValueError(f"不支持的应用别名：{app_name}；支持 {', '.join(APP_CANDIDATES)}")
    candidates = APP_CANDIDATES[normalized]
    path = next((item for item in candidates if item.exists()), None)
    if path is None and normalized == "idea":
        path = _find_jetbrains_idea()
    if path is None:
        raise FileNotFoundError(f"未找到 {app_name}，请检查安装路径")
    process = subprocess.Popen([str(path)], shell=False)
    return {"launched": True, "app": normalized, "pid": process.pid, "path": str(path)}


def open_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("只允许打开有效的 http/https URL")
    opened = webbrowser.open(url, new=2)
    return {"opened": bool(opened), "url": url}


from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Any

import psutil

from agent.security import SecurityPolicy
from tools.apps import AppCatalog


class ProcessManager:
    def __init__(self, security: SecurityPolicy, catalog: AppCatalog) -> None:
        self.security = security
        self.catalog = catalog
        self._managed: dict[int, subprocess.Popen[str]] = {}
        self._lock = threading.Lock()

    def start_process(self, program: str, args: list[str]) -> dict[str, Any]:
        recipe = self.catalog.resolve(program, args)
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if recipe.new_console:
            creationflags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        stream_target = None if recipe.new_console else subprocess.DEVNULL
        process = subprocess.Popen(
            list(recipe.command),
            cwd=recipe.cwd,
            shell=False,
            text=True,
            stdin=stream_target,
            stdout=stream_target,
            stderr=stream_target,
            creationflags=creationflags,
        )
        with self._lock:
            self._managed[process.pid] = process
        return {
            "pid": process.pid,
            "program": recipe.name,
            "executable": str(recipe.executable),
            "cwd": str(recipe.cwd),
        }

    def launch_app(self, app_name: str) -> dict[str, Any]:
        result = self.start_process(app_name, [])
        return {"launched": True, "app": result["program"], **result}

    def is_app_running(self, app_name: str) -> dict[str, Any]:
        canonical = self.catalog.canonical_name(app_name)
        names = set(self.catalog.process_names(canonical))
        pids: list[int] = []
        matched_names: list[str] = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = str(proc.info.get("name") or "")
                if name.casefold() in names:
                    pids.append(int(proc.info["pid"]))
                    matched_names.append(name)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return {
            "app": canonical,
            "running": bool(pids),
            "pids": pids,
            "process_names": list(dict.fromkeys(matched_names)),
        }

    def focus_app(self, app_name: str) -> dict[str, Any]:
        state = self.is_app_running(app_name)
        if not state["running"]:
            raise RuntimeError(f"应用未运行：{state['app']}")

        from pywinauto import Desktop

        pid_set = set(state["pids"])
        candidates = []
        for window in Desktop(backend="uia").windows():
            try:
                process_id = int(window.element_info.process_id)
                if process_id not in pid_set or not window.is_visible():
                    continue
                title = window.window_text().strip()
                candidates.append((window, title))
            except Exception:
                continue

        if not candidates:
            raise RuntimeError(f"应用正在运行，但没有找到可置前窗口：{state['app']}")

        # 优先选择有标题的顶层窗口，避免把隐藏辅助窗口切到前台。
        window, title = next(((w, t) for w, t in candidates if t), candidates[0])
        try:
            if window.is_minimized():
                window.restore()
        except Exception:
            pass
        window.set_focus()
        return {
            "app": state["app"],
            "focused": True,
            "title": title,
            "pid": int(window.element_info.process_id),
        }

    def close_app(self, app_name: str) -> dict[str, Any]:
        state = self.is_app_running(app_name)
        if not state["running"]:
            return {"app": state["app"], "closed": False, "already_stopped": True, "pids": []}

        targets = []
        for pid in state["pids"]:
            try:
                targets.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
        for proc in targets:
            try:
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _, alive = psutil.wait_procs(targets, timeout=5)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return {"app": state["app"], "closed": True, "pids": state["pids"]}

    def stop_managed_process(self, pid: int) -> dict[str, Any]:
        with self._lock:
            process = self._managed.get(pid)
        if process is None:
            raise PermissionError("只能停止由 Agent 启动并管理的进程")
        if process.poll() is not None:
            with self._lock:
                self._managed.pop(pid, None)
            return {"stopped": False, "already_exited": True, "pid": pid}
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            with self._lock:
                self._managed.pop(pid, None)
            return {"stopped": False, "already_exited": True, "pid": pid}
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=5)
        for item in alive:
            item.kill()
        with self._lock:
            self._managed.pop(pid, None)
        return {"stopped": True, "pid": pid}

    # 兼容现有内部调用名称；不向 AI 暴露。
    stop_process = stop_managed_process

    def list_managed(self) -> dict[str, Any]:
        rows = []
        with self._lock:
            for pid, process in list(self._managed.items()):
                if process.poll() is not None:
                    self._managed.pop(pid, None)
                    continue
                rows.append({"pid": pid, "command": process.args})
        return {"processes": rows, "count": len(rows)}

    def run_safe_command(
        self, command: list[str], cwd: str | None = None, timeout: int = 60
    ) -> dict[str, Any]:
        working_dir = self.security.resolve_allowed_path(cwd) if cwd else self.security.settings.base_dir
        self.security.validate_command(command, working_dir)
        completed = subprocess.run(
            command,
            cwd=working_dir,
            shell=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(1, min(timeout, 300)),
            check=False,
        )
        limit = 20_000
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-limit:],
            "stderr": completed.stderr[-limit:],
            "truncated": len(completed.stdout) > limit or len(completed.stderr) > limit,
        }


def list_processes() -> dict[str, Any]:
    rows = []
    for proc in psutil.process_iter(["pid", "name", "status", "cmdline"]):
        try:
            info = proc.info
            command_parts = info.get("cmdline") or []
            script = None
            if str(info.get("name") or "").lower().startswith("python") and len(command_parts) > 1:
                candidate = str(command_parts[1])
                if candidate.lower().endswith(".py"):
                    script = candidate[:300]
            rows.append(
                {
                    "pid": info["pid"],
                    "name": info.get("name"),
                    "status": info.get("status"),
                    "script": script,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if len(rows) >= 120:
            break
    return {"processes": rows, "count": len(rows)}

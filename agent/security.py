from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import Settings


SHELL_META = re.compile(r"[|&;<>`\n\r]")


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    requires_confirmation: bool
    reason: str = ""
    description: str = ""


class SecurityPolicy:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def assess(self, tool_name: str, arguments: dict[str, Any]) -> RiskAssessment:
        if tool_name == "close_app":
            app_name = str(arguments.get("app_name", "")).strip() or "未知应用"
            return RiskAssessment(True, "关闭应用可能导致未保存内容丢失", f"关闭应用：{app_name}")

        command = arguments.get("command", [])
        tokens = [str(token).lower() for token in command] if isinstance(command, list) else []
        joined = " ".join(tokens)
        if tool_name == "run_safe_command":
            risky = (
                "git push" in joined
                or "git clean" in joined
                or ("git reset" in joined and "--hard" in tokens)
                or any(word in joined for word in ("shutdown", "reboot", "format"))
            )
            if risky:
                return RiskAssessment(True, "高风险命令", f"运行命令：{joined}")
        return RiskAssessment(False)

    def validate_command(self, command: list[str], cwd: str | Path | None = None) -> None:
        if not command or not all(isinstance(item, str) and item.strip() for item in command):
            raise ValueError("command 必须是非空字符串数组")
        if any(SHELL_META.search(item) for item in command):
            raise ValueError("命令包含禁止的 Shell 元字符")
        executable = Path(command[0]).name.lower()
        if executable not in self.settings.allowed_executables:
            raise PermissionError(f"程序不在白名单中：{executable}")
        if executable in {"python", "python.exe", "pythonw.exe"}:
            lowered = {item.lower() for item in command[1:]}
            if "-c" in lowered or "-m" in lowered:
                raise PermissionError("禁止通过 python -c/-m 执行动态代码或模块")
            if len(command) < 2 or command[1].startswith("-") or not command[1].lower().endswith(".py"):
                raise PermissionError("Python 只能启动允许目录中的 .py 文件")
            base = Path(cwd).resolve() if cwd else self.settings.base_dir
            script = Path(command[1])
            self.resolve_allowed_path(str(script if script.is_absolute() else base / script))
        if executable in {"git", "git.exe"}:
            if len(command) < 2 or command[1].startswith("-"):
                raise PermissionError("Git 必须直接使用允许的子命令，禁止全局选项")
            allowed_git = {"status", "log", "diff", "branch", "fetch", "pull", "push", "reset", "clean"}
            if command[1].lower() not in allowed_git:
                raise PermissionError(f"Git 子命令不在白名单中：{command[1]}")

    def resolve_allowed_path(self, raw_path: str, *, must_exist: bool = True) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if not any(path == root or root in path.parents for root in self.settings.allowed_roots):
            raise PermissionError(f"路径不在 AGENT_ALLOWED_ROOTS 中：{path}")
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path

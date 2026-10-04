from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class AppSpec:
    name: str
    paths: tuple[Path, ...]
    cwd: Path | None
    default_args: tuple[str, ...]
    allow_extra_args: bool
    new_console: bool


@dataclass(frozen=True, slots=True)
class LaunchRecipe:
    name: str
    executable: Path
    command: tuple[str, ...]
    cwd: Path
    new_console: bool


class AppCatalog:
    """从受信任的本地 YAML 读取应用配方；模型只能引用应用名称。"""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self._apps = self._load()

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._apps))

    def resolve(self, program: str, args: list[str] | None = None) -> LaunchRecipe:
        name = program.strip().lower()
        spec = self._apps.get(name)
        if spec is None:
            raise PermissionError(f"应用不在白名单中：{program}；允许：{', '.join(self.names())}")
        extra = args or []
        if not isinstance(extra, list) or any(not isinstance(value, str) for value in extra):
            raise TypeError("args 必须是字符串数组")
        if extra and not spec.allow_extra_args:
            raise PermissionError(f"应用 {name} 不允许 AI 传入额外参数")
        executable = next((path for path in spec.paths if path.is_file()), None)
        if executable is None:
            candidates = ", ".join(str(path) for path in spec.paths)
            raise FileNotFoundError(f"未找到应用 {name}，已检查：{candidates}")
        cwd = spec.cwd or executable.parent
        if not cwd.is_dir():
            raise NotADirectoryError(f"应用工作目录不存在：{cwd}")
        command = (str(executable), *spec.default_args, *extra)
        return LaunchRecipe(name, executable, command, cwd, spec.new_console)

    def _load(self) -> dict[str, AppSpec]:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"应用配置不存在：{self.config_path}")
        document = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        raw_apps = document.get("apps")
        if not isinstance(raw_apps, dict) or not raw_apps:
            raise ValueError("apps.yaml 必须包含非空 apps 映射")
        apps: dict[str, AppSpec] = {}
        for raw_name, raw_spec in raw_apps.items():
            name = str(raw_name).strip().lower()
            if not name or not isinstance(raw_spec, dict):
                raise ValueError("应用名称和配置必须有效")
            raw_paths = raw_spec.get("paths")
            if not isinstance(raw_paths, list) or not raw_paths:
                raise ValueError(f"应用 {name} 缺少 paths")
            paths = tuple(self._expand_path(value) for value in raw_paths)
            cwd_value = raw_spec.get("cwd")
            cwd = self._expand_path(cwd_value) if cwd_value else None
            default_args = raw_spec.get("default_args", [])
            if not isinstance(default_args, list) or any(not isinstance(value, str) for value in default_args):
                raise ValueError(f"应用 {name} 的 default_args 必须是字符串数组")
            apps[name] = AppSpec(
                name=name,
                paths=paths,
                cwd=cwd,
                default_args=tuple(default_args),
                allow_extra_args=bool(raw_spec.get("allow_extra_args", False)),
                new_console=bool(raw_spec.get("new_console", False)),
            )
        return apps

    @staticmethod
    def _expand_path(value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("应用路径必须是非空字符串")
        return Path(os.path.expandvars(value)).expanduser().resolve()


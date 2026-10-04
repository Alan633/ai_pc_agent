from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_SHARED_INTERPRETER_NAMES = {
    "python",
    "python.exe",
    "python3",
    "python3.exe",
    "pythonw",
    "pythonw.exe",
}


@dataclass(frozen=True, slots=True)
class AppSpec:
    name: str
    aliases: tuple[str, ...]
    process_names: tuple[str, ...]
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
    """从受信任的本地 YAML 读取应用配方；模型只能引用应用名称或显式别名。"""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self._apps, self._aliases = self._load()

    def names(self) -> tuple[str, ...]:
        """返回模型可直接使用的规范名和别名。"""
        return tuple(sorted(self._aliases))

    def canonical_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._apps))

    def canonical_name(self, program: str) -> str:
        key = program.strip().casefold()
        name = self._aliases.get(key)
        if name is None:
            raise PermissionError(f"应用不在白名单中：{program}；允许：{', '.join(self.names())}")
        return name

    def process_names(self, program: str) -> tuple[str, ...]:
        spec = self._apps[self.canonical_name(program)]
        configured = tuple(name.casefold() for name in spec.process_names if name.strip())
        derived = tuple(path.name.casefold() for path in spec.paths if path.name)
        return tuple(dict.fromkeys((*configured, *derived)))

    def matches_process(
        self,
        program: str,
        *,
        name: str,
        executable: str | None = None,
        command_line: list[str] | tuple[str, ...] | None = None,
    ) -> bool:
        """判断系统进程是否确实属于白名单应用。

        普通桌面应用可按其专用进程名识别；Python 这类共享解释器必须同时
        匹配白名单解释器路径和默认启动参数，避免把其他脚本当成目标应用。
        """
        spec = self._apps[self.canonical_name(program)]
        process_name = name.strip().casefold()
        if process_name not in self.process_names(spec.name):
            return False
        if process_name not in _SHARED_INTERPRETER_NAMES:
            return True
        if not executable or not spec.default_args:
            return False

        allowed_executables = {self._normalize_path(path) for path in spec.paths}
        if self._normalize_path(executable) not in allowed_executables:
            return False

        actual_arguments = {
            self._normalize_argument(value) for value in (command_line or ())[1:]
        }
        required_arguments = {
            self._normalize_argument(value) for value in spec.default_args
        }
        return required_arguments.issubset(actual_arguments)

    def resolve(self, program: str, args: list[str] | None = None) -> LaunchRecipe:
        name = self.canonical_name(program)
        spec = self._apps[name]
        extra = args or []
        if not isinstance(extra, list) or any(not isinstance(value, str) for value in extra):
            raise TypeError("args 必须是字符串数组")
        if extra and not spec.allow_extra_args:
            raise PermissionError(f"应用 {name} 不允许 AI 传入额外参数")
        executable = next((path for path in spec.paths if path.is_file()), None)
        if executable is None:
            candidates = ", ".join(str(path) for path in spec.paths)
            raise FileNotFoundError(
                f"未找到应用 {name}。已检查：{candidates}。"
                "如应用安装在自定义目录，请只修改 config/apps.yaml 的白名单路径。"
            )
        cwd = spec.cwd or executable.parent
        if not cwd.is_dir():
            raise NotADirectoryError(f"应用工作目录不存在：{cwd}")
        command = (str(executable), *spec.default_args, *extra)
        return LaunchRecipe(name, executable, command, cwd, spec.new_console)

    def _load(self) -> tuple[dict[str, AppSpec], dict[str, str]]:
        if not self.config_path.is_file():
            raise FileNotFoundError(f"应用配置不存在：{self.config_path}")
        document = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        raw_apps = document.get("apps")
        if not isinstance(raw_apps, dict) or not raw_apps:
            raise ValueError("apps.yaml 必须包含非空 apps 映射")

        apps: dict[str, AppSpec] = {}
        aliases: dict[str, str] = {}
        for raw_name, raw_spec in raw_apps.items():
            name = str(raw_name).strip().casefold()
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

            raw_aliases = raw_spec.get("aliases", [])
            if not isinstance(raw_aliases, list) or any(not isinstance(value, str) for value in raw_aliases):
                raise ValueError(f"应用 {name} 的 aliases 必须是字符串数组")
            app_aliases = tuple(value.strip().casefold() for value in raw_aliases if value.strip())

            raw_process_names = raw_spec.get("process_names", [])
            if not isinstance(raw_process_names, list) or any(
                not isinstance(value, str) for value in raw_process_names
            ):
                raise ValueError(f"应用 {name} 的 process_names 必须是字符串数组")
            process_names = tuple(value.strip() for value in raw_process_names if value.strip())

            spec = AppSpec(
                name=name,
                aliases=app_aliases,
                process_names=process_names,
                paths=paths,
                cwd=cwd,
                default_args=tuple(default_args),
                allow_extra_args=bool(raw_spec.get("allow_extra_args", False)),
                new_console=bool(raw_spec.get("new_console", False)),
            )
            apps[name] = spec

            for alias in (name, *app_aliases):
                if alias in aliases and aliases[alias] != name:
                    raise ValueError(f"应用别名重复：{alias}")
                aliases[alias] = name

        return apps, aliases

    @staticmethod
    def _expand_path(value: Any) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("应用路径必须是非空字符串")
        return Path(os.path.expandvars(value)).expanduser().resolve()

    @staticmethod
    def _normalize_path(value: str | Path) -> str:
        return os.path.normcase(os.path.normpath(str(Path(value).expanduser().resolve())))

    @classmethod
    def _normalize_argument(cls, value: str) -> str:
        text = str(value).strip()
        if Path(text).is_absolute():
            return cls._normalize_path(text)
        return text.casefold()

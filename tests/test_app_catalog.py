from __future__ import annotations

from pathlib import Path

from tools.apps import AppCatalog


def test_app_alias_and_process_name(tmp_path: Path) -> None:
    executable = tmp_path / "Weixin.exe"
    executable.write_bytes(b"")
    config = tmp_path / "apps.yaml"
    config.write_text(
        f"""
apps:
  wechat:
    paths:
      - "{executable.as_posix()}"
    aliases:
      - "微信"
      - "weixin"
    process_names:
      - "Weixin.exe"
    allow_extra_args: false
""".strip(),
        encoding="utf-8",
    )

    catalog = AppCatalog(config)
    assert catalog.canonical_name("微信") == "wechat"
    assert catalog.canonical_name("WEIXIN") == "wechat"
    assert "微信" in catalog.names()
    assert "weixin.exe" in catalog.process_names("wechat")
    assert catalog.resolve("微信").name == "wechat"


def test_shared_python_process_requires_executable_and_script_match(tmp_path: Path) -> None:
    executable = tmp_path / ".venv" / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    main_script = tmp_path / "main.py"
    main_script.write_text("", encoding="utf-8")
    other_script = tmp_path / "other.py"
    other_script.write_text("", encoding="utf-8")
    config = tmp_path / "apps.yaml"
    config.write_text(
        f"""
apps:
  stock_app:
    paths:
      - "{executable.as_posix()}"
    default_args:
      - "{main_script.as_posix()}"
    process_names:
      - "python.exe"
""".strip(),
        encoding="utf-8",
    )

    catalog = AppCatalog(config)
    assert catalog.matches_process(
        "stock_app",
        name="python.exe",
        executable=str(executable),
        command_line=[str(executable), str(main_script)],
    )
    assert not catalog.matches_process(
        "stock_app",
        name="python.exe",
        executable=str(executable),
        command_line=[str(executable), str(other_script)],
    )
    assert not catalog.matches_process(
        "stock_app",
        name="python.exe",
        executable=str(tmp_path / "other-venv" / "python.exe"),
        command_line=[str(executable), str(main_script)],
    )

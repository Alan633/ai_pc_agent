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

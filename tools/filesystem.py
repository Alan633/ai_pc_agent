from __future__ import annotations

from typing import Any

from agent.security import SecurityPolicy


def read_text_file(path: str, security: SecurityPolicy, max_chars: int = 50_000) -> dict[str, Any]:
    resolved = security.resolve_allowed_path(path)
    if not resolved.is_file():
        raise IsADirectoryError(resolved)
    limit = max(1, min(max_chars, 200_000))
    content = resolved.read_text(encoding="utf-8", errors="replace")
    return {"path": str(resolved), "content": content[:limit], "truncated": len(content) > limit}


def list_directory(path: str, security: SecurityPolicy, limit: int = 200) -> dict[str, Any]:
    resolved = security.resolve_allowed_path(path)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    entries = []
    for child in sorted(resolved.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        entries.append({"name": child.name, "type": "directory" if child.is_dir() else "file"})
        if len(entries) >= max(1, min(limit, 1000)):
            break
    return {"path": str(resolved), "entries": entries}


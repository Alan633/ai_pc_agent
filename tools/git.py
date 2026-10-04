from __future__ import annotations

import subprocess
from typing import Any

from agent.security import SecurityPolicy


def git_pull(repo_path: str, security: SecurityPolicy) -> dict[str, Any]:
    repo = security.resolve_allowed_path(repo_path)
    if not (repo / ".git").exists():
        raise ValueError(f"不是 Git 仓库：{repo}")
    completed = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=repo,
        shell=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout[-20_000:],
        "stderr": completed.stderr[-20_000:],
    }


from __future__ import annotations

from typing import Any


def get_active_window() -> dict[str, Any]:
    from pywinauto import Desktop

    window = Desktop(backend="uia").get_active()
    return {
        "title": window.window_text(),
        "class_name": window.class_name(),
        "pid": int(window.element_info.process_id),
    }


def list_windows() -> dict[str, Any]:
    """列出当前可见顶层窗口，供模型做确定性窗口选择。"""
    from pywinauto import Desktop

    rows: list[dict[str, Any]] = []
    for window in Desktop(backend="uia").windows():
        try:
            if not window.is_visible():
                continue
            title = window.window_text().strip()
            if not title:
                continue
            rows.append(
                {
                    "title": title[:200],
                    "class_name": window.class_name(),
                    "pid": int(window.element_info.process_id),
                    "minimized": bool(window.is_minimized()),
                }
            )
        except Exception:
            continue
        if len(rows) >= 80:
            break
    return {"windows": rows, "count": len(rows)}


def focus_window(title_keyword: str) -> dict[str, Any]:
    """按标题关键字把已存在的顶层窗口置前，不依赖鼠标坐标。"""
    keyword = title_keyword.strip().casefold()
    if not keyword:
        raise ValueError("title_keyword 不能为空")

    from pywinauto import Desktop

    matches = []
    for window in Desktop(backend="uia").windows():
        try:
            title = window.window_text().strip()
            if not title or keyword not in title.casefold():
                continue
            matches.append((window, title))
        except Exception:
            continue

    if not matches:
        raise LookupError(f"没有找到标题包含“{title_keyword}”的窗口")

    window, title = matches[0]
    try:
        if window.is_minimized():
            window.restore()
    except Exception:
        pass
    window.set_focus()
    return {
        "focused": True,
        "title": title,
        "class_name": window.class_name(),
        "pid": int(window.element_info.process_id),
    }


def _find_element(
    window_title_re: str,
    title: str | None = None,
    auto_id: str | None = None,
    control_type: str | None = None,
) -> Any:
    from pywinauto import Desktop

    window = Desktop(backend="uia").window(title_re=window_title_re)
    window.wait("visible", timeout=10)
    criteria = {
        key: value
        for key, value in {
            "title": title,
            "auto_id": auto_id,
            "control_type": control_type,
        }.items()
        if value
    }
    if not criteria:
        raise ValueError("至少提供 title、auto_id、control_type 之一")
    element = window.child_window(**criteria)
    element.wait("exists enabled visible", timeout=10)
    return element


def uia_find_element(
    window_title_re: str,
    title: str | None = None,
    auto_id: str | None = None,
    control_type: str | None = None,
) -> dict[str, Any]:
    element = _find_element(window_title_re, title, auto_id, control_type)
    rectangle = element.rectangle()
    return {
        "found": True,
        "title": element.window_text(),
        "control_type": element.element_info.control_type,
        "rectangle": [rectangle.left, rectangle.top, rectangle.right, rectangle.bottom],
    }


def uia_click(
    window_title_re: str,
    title: str | None = None,
    auto_id: str | None = None,
    control_type: str | None = None,
) -> dict[str, Any]:
    element = _find_element(window_title_re, title, auto_id, control_type)
    element.click_input()
    return {"clicked": True, "title": element.window_text()}


def uia_type(
    window_title_re: str,
    text: str,
    title: str | None = None,
    auto_id: str | None = None,
    control_type: str | None = None,
) -> dict[str, Any]:
    element = _find_element(window_title_re, title, auto_id, control_type)
    try:
        element.set_edit_text(text)
    except AttributeError:
        element.type_keys(text, with_spaces=True, set_foreground=True)
    return {"typed": True, "length": len(text)}

from __future__ import annotations

from typing import Any


def get_active_window() -> dict[str, Any]:
    from pywinauto import Desktop

    window = Desktop(backend="uia").get_active()
    return {"title": window.window_text(), "class_name": window.class_name()}


def _find_element(
    window_title_re: str,
    title: str | None = None,
    auto_id: str | None = None,
    control_type: str | None = None,
) -> Any:
    from pywinauto import Desktop

    window = Desktop(backend="uia").window(title_re=window_title_re)
    window.wait("visible", timeout=10)
    criteria = {key: value for key, value in {
        "title": title, "auto_id": auto_id, "control_type": control_type
    }.items() if value}
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


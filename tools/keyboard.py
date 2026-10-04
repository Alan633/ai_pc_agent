from __future__ import annotations

import pyautogui
import pyperclip


ALLOWED_KEYS = frozenset(pyautogui.KEYBOARD_KEYS)


def type_text(text: str, interval: float = 0.02) -> dict[str, object]:
    if len(text) > 10_000:
        raise ValueError("单次输入文本不能超过 10000 字符")
    if any(ord(character) > 127 for character in text):
        # PyAutoGUI.write 不支持 Unicode；临时使用剪贴板粘贴并尽量恢复原内容。
        previous = None
        try:
            previous = pyperclip.paste()
        except pyperclip.PyperclipException:
            pass
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        if previous is not None:
            pyperclip.copy(previous)
    else:
        pyautogui.write(text, interval=max(0.0, min(interval, 1.0)))
    return {"typed": True, "length": len(text)}


def press_key(key: str) -> dict[str, str]:
    normalized = key.lower()
    if normalized not in ALLOWED_KEYS:
        raise ValueError(f"不支持的按键：{key}")
    pyautogui.press(normalized)
    return {"pressed": normalized}


def hotkey(keys: list[str]) -> dict[str, list[str]]:
    normalized = [key.lower() for key in keys]
    if not 2 <= len(normalized) <= 5 or any(key not in ALLOWED_KEYS for key in normalized):
        raise ValueError("组合键必须包含 2-5 个有效按键")
    pyautogui.hotkey(*normalized)
    return {"hotkey": normalized}

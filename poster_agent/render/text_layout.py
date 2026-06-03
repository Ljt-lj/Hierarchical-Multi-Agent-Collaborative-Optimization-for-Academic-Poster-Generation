"""像素级文本换行，防止超出边框."""

from __future__ import annotations


def text_width(text: str, font) -> int:
    try:
        return int(font.getlength(text))
    except Exception:
        return len(text) * getattr(font, "size", 12)


def wrap_to_width(text: str, font, max_width: int) -> list[str]:
    if max_width < 20 or not text:
        return [text] if text else []
    words = text.split()
    if not words:
        return [text]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if text_width(trial, font) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines or [text]


def truncate_to_width(text: str, font, max_width: int, suffix: str = "...") -> str:
    if text_width(text, font) <= max_width:
        return text
    trimmed = text
    while trimmed and text_width(trimmed + suffix, font) > max_width:
        trimmed = trimmed[:-1]
    return (trimmed + suffix) if trimmed else suffix

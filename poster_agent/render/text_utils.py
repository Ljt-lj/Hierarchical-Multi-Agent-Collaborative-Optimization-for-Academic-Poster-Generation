"""文本格式化：Title Case、可读性处理."""

from __future__ import annotations

import re


def smart_title(text: str) -> str:
    """全大写英文转首字母大写，保留中文与其它语言."""
    if not text or not text.strip():
        return text
    if _is_mostly_uppercase(text):
        return _to_title_case(text)
    return text


def format_body(text: str) -> str:
    """正文格式化：修正全大写片段."""
    if not text:
        return text
    if _is_mostly_uppercase(text):
        return _to_title_case(text)
    # 修正连续全大写单词
    def repl(m: re.Match) -> str:
        word = m.group(0)
        if len(word) > 2 and word.isupper():
            return word.capitalize()
        return word

    return re.sub(r"[A-Za-z]{3,}", repl, text)


def _is_mostly_uppercase(text: str) -> bool:
    letters = [c for c in text if c.isalpha() and ord(c) < 128]
    if len(letters) < 3:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.75


def _to_title_case(text: str) -> str:
    small = {"a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with", "by"}
    words = re.split(r"(\s+)", text.strip())
    out: list[str] = []
    word_idx = 0
    for w in words:
        if not w.strip():
            out.append(w)
            continue
        if re.search(r"[\u4e00-\u9fff]", w):
            out.append(w)
        elif word_idx == 0 or w.lower() not in small:
            out.append(w.capitalize() if w.isupper() else w.title())
        else:
            out.append(w.lower())
        word_idx += 1
    return "".join(out)

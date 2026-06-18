"""文本清洗：去除 PDF 解析残留."""

from __future__ import annotations

import re


_PAGE_MARK = re.compile(r"---\s*Page\s*\d+\s*---", re.I)
_URL = re.compile(r"https?://\S+")
_DOI = re.compile(r"doi\.org/\S+", re.I)
_JOURNAL_HEADER = re.compile(
    r"^(Nature Methods|nature methods|Article|\d{4,5})\s*$",
    re.I | re.M,
)


def sanitize_poster_text(text: str) -> str:
    if not text:
        return ""
    text = _PAGE_MARK.sub("", text)
    text = _URL.sub("", text)
    text = _DOI.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sanitize_bullet(text: str) -> str:
    text = sanitize_poster_text(text)
    if len(text) > 280:
        text = text[:277] + "..."
    return text


def condense_bullet(text: str, *, max_words: int = 22) -> str:
    text = sanitize_bullet(text)
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def condense_summary(text: str, *, max_words: int = 30) -> str:
    text = sanitize_poster_text(text)
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(",;:") + "..."


def is_garbage_bullet(text: str) -> bool:
    t = text.strip()
    if not t or len(t) < 12:
        return True
    if _PAGE_MARK.search(t):
        return True
    if "nature methods" in t.lower() and len(t) > 120:
        return True
    if t.count("\n") > 8:
        return True
    return False


def clean_bullets(bullets: list[str], *, min_keep: int = 3, max_keep: int = 5) -> list[str]:
    cleaned: list[str] = []
    for b in bullets:
        s = sanitize_bullet(str(b))
        if is_garbage_bullet(s):
            continue
        cleaned.append(s)
    return cleaned[:max_keep] if cleaned else [sanitize_bullet(b) for b in bullets[:min_keep] if b.strip()]

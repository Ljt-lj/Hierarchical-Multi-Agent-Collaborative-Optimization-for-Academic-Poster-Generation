"""论文语言检测."""

from __future__ import annotations

import re

from poster_agent.models.trees import RawNode


def detect_language(text: str) -> str:
    if not text:
        return "en"
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if cjk > max(latin * 0.25, 20):
        return "zh"
    return "en"


def detect_from_raw_tree(raw: RawNode) -> str:
    parts = [raw.title, raw.content]

    def walk(n: RawNode) -> None:
        parts.append(n.title)
        parts.append(n.content)
        for c in n.children:
            walk(c)

    for c in raw.children:
        walk(c)
    return detect_language(" ".join(parts))


def language_instruction(lang: str) -> str:
    if lang == "zh":
        return "语言要求：全文使用中文，与论文语言保持一致，不要翻译为英文。"
    return (
        "Language requirement: Keep ALL output in English, matching the paper language. "
        "Do NOT translate to Chinese. Preserve original terminology and section titles."
    )

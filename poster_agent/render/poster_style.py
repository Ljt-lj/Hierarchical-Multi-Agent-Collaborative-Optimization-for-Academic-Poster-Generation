"""学术海报统一视觉风格（对标 Nature/本科优秀海报：橙棕强调 + 白底流程图 + 统一表格）."""

from __future__ import annotations

import re
from pathlib import Path

# 主强调色 — 与 ORF3a 参考海报一致的 burnt orange
ACCENT_RGB = (196, 92, 38)
ACCENT_HEX = "#C45C26"
ACCENT_DARK = (160, 72, 28)
ACCENT_LIGHT = (245, 236, 228)

BG_CANVAS = (250, 250, 250)
BG_SECTION = (255, 255, 255)
BG_FIGURE_PANEL = (255, 255, 255)
BG_TABLE_ALT = (245, 245, 245)

TEXT_PRIMARY = (28, 28, 28)
TEXT_SECONDARY = (55, 55, 55)
TEXT_MUTED = (100, 100, 100)
TEXT_CAPTION = (85, 85, 85)

BORDER_LIGHT = (210, 210, 210)
BORDER_MEDIUM = (160, 160, 160)
ARROW_GRAY = (100, 100, 100)

# 图表：强调色 + 中性灰（避免彩虹色）
CHART_ACCENT = ACCENT_HEX
CHART_NEUTRAL = "#9A9A9A"
CHART_PALETTE = [CHART_ACCENT, "#8E8E8E", "#B5B5B5", "#757575", "#C4A882"]

FLOW_FILL = (255, 255, 255)
FLOW_BORDER = (40, 40, 40)
FLOW_TITLE = TEXT_PRIMARY


def chart_colors(labels: list[str], *, highlight: str = "grasp") -> list[str]:
    """最佳方法用强调色，其余用灰."""
    hl = highlight.lower()
    return [
        CHART_ACCENT if hl in str(lbl).lower() else CHART_NEUTRAL
        for lbl in labels
    ]


def figure_panel_title(caption: str, path: str = "", fallback: str = "") -> str:
    """从 caption 提取插图上方展示的描述性小标题（对标 ORF3a 参考海报）."""
    if fallback:
        return fallback[:90]
    t = clean_figure_caption(caption, path)
    t = re.sub(r"^Fig\.?\s*\d+[a-z]?\s*:?\s*", "", t, flags=re.I).strip()
    t = re.sub(r"^(and|see|supplementary)\s+", "", t, flags=re.I).strip()
    if len(t) < 12:
        stem = Path(path).stem if path else ""
        m = re.search(r"fig(\d+)", stem, re.I)
        if m:
            return f"Figure {m.group(1)} Results"
        return fallback or "Results Overview"
    if len(t) > 90:
        t = t[:87].rstrip() + "…"
    return t[0].upper() + t[1:] if t else fallback


def clean_figure_caption(text: str, path: str = "") -> str:
    """规范化论文 Fig  caption，去掉内部 tag."""
    if not text:
        stem = Path(path).stem if path else ""
        m = re.search(r"fig(\d+)", stem, re.I)
        return f"Fig. {m.group(1)}" if m else ""
    t = re.sub(r"\s*\[[^\]]+\]\s*", " ", text).strip()
    t = re.sub(r"\s+", " ", t)
    if not re.match(r"^Fig\.?\s*\d", t, re.I):
        stem = Path(path).stem if path else ""
        m = re.search(r"fig(\d+)", stem, re.I)
        if m:
            prefix = f"Fig. {m.group(1)}:"
            t = f"{prefix} {t}" if t else prefix
    if len(t) > 110:
        t = t[:107].rstrip() + "…"
    return t

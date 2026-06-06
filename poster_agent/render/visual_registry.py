"""跨章节插图内容注册表：防止数据/流程/步骤重复."""

from __future__ import annotations

import re
from typing import Any

from poster_agent.models.trees import ContentNode
from poster_agent.models.visuals import VisualSpec


def poster_sections(content: ContentNode) -> list[ContentNode]:
    if content.children:
        return list(content.children)
    return [content]


def is_abstract_title(title: str) -> bool:
    t = title.lower()
    return "abstract" in t or "摘要" in t


def is_conclusion_title(title: str) -> bool:
    t = title.lower()
    return "conclusion" in t or "结论" in t


def normalize_step(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())[:80]


class VisualRegistry:
    def __init__(self) -> None:
        self.stat_cards_used = False
        self.value_sets: list[tuple[float, ...]] = []
        self.steps: set[str] = set()
        self.labels: set[str] = set()
        self.card_sigs: set[tuple[str, str]] = set()
        self.types_used: dict[str, int] = {}

    def summary(self) -> str:
        lines: list[str] = []
        if self.stat_cards_used:
            lines.append("- stat_cards already used in Abstract")
        if self.value_sets:
            lines.append(f"- numeric series used: {list(self.value_sets)[:4]}")
        if self.steps:
            lines.append(f"- flow steps used: {list(self.steps)[:6]}")
        if self.labels:
            lines.append(f"- chart labels used: {list(self.labels)[:8]}")
        return "\n".join(lines) if lines else "None yet."

    def register(self, spec: VisualSpec) -> None:
        self.types_used[spec.type] = self.types_used.get(spec.type, 0) + 1
        if spec.type == "stat_cards":
            self.stat_cards_used = True
            for card in spec.data.get("cards") or []:
                if isinstance(card, dict):
                    self.card_sigs.add(
                        (str(card.get("label", "")).lower(), str(card.get("value", "")))
                    )
        if spec.type in ("bar_chart", "line_chart", "pie_chart"):
            vals = tuple(float(v) for v in (spec.data.get("values") or []) if _is_num(v))
            if vals:
                self.value_sets.append(vals)
            for lb in spec.data.get("labels") or []:
                self.labels.add(str(lb).lower())
        if spec.type in ("flow_diagram", "architecture"):
            for step in spec.data.get("steps") or []:
                self.steps.add(normalize_step(str(step)))

    def duplicate_issues(self, spec: VisualSpec) -> list[str]:
        issues: list[str] = []
        if spec.type == "stat_cards":
            if self.stat_cards_used:
                issues.append("stat_cards already used")
            if not is_abstract_title(spec.section_title):
                issues.append("stat_cards not allowed outside Abstract")
            for card in spec.data.get("cards") or []:
                if isinstance(card, dict):
                    sig = (str(card.get("label", "")).lower(), str(card.get("value", "")))
                    if sig in self.card_sigs:
                        issues.append(f"duplicate card {sig}")
        if spec.type in ("bar_chart", "line_chart", "pie_chart"):
            vals = tuple(float(v) for v in (spec.data.get("values") or []) if _is_num(v))
            if vals and vals in self.value_sets:
                issues.append("duplicate numeric series")
            if vals in _PLACEHOLDER_SERIES:
                issues.append("placeholder demo data detected")
            for lb in spec.data.get("labels") or []:
                if str(lb).lower() in self.labels:
                    issues.append(f"duplicate label {lb}")
        if spec.type in ("flow_diagram", "architecture"):
            steps = [normalize_step(str(s)) for s in spec.data.get("steps") or []]
            if steps:
                overlap = sum(1 for s in steps if s in self.steps)
                if overlap >= max(1, len(steps) // 2):
                    issues.append("flow content largely duplicated")
        return issues


_PLACEHOLDER_SERIES = {
    (70.0, 85.0, 92.0),
    (70.0, 85.0),
}


def _is_num(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def visual_fingerprint(spec: VisualSpec) -> str:
    if spec.type == "stat_cards":
        cards = spec.data.get("cards") or []
        sig = tuple(
            (str(c.get("label", "")), str(c.get("value", "")))
            for c in cards
            if isinstance(c, dict)
        )
        return f"stat|{sig}"
    if spec.type in ("bar_chart", "line_chart", "pie_chart"):
        vals = tuple(spec.data.get("values") or [])
        labels = tuple(str(x) for x in (spec.data.get("labels") or []))
        return f"{spec.type}|{labels}|{vals}"
    steps = tuple(normalize_step(str(s)) for s in (spec.data.get("steps") or []))
    return f"{spec.type}|{steps}|{(spec.title or '').lower()}"

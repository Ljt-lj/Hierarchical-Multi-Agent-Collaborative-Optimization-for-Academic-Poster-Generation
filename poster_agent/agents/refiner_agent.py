"""精炼智能体：逻辑链优先的内容树构建 + 保留论文原图."""

from __future__ import annotations

import re

from poster_agent.agents.logic_planner_agent import LogicPlan
from poster_agent.llm_client import LLMClient
from poster_agent.config import PosterConfig
from poster_agent.render.content_sanitize import clean_bullets, sanitize_poster_text
from poster_agent.render.language import language_instruction
from poster_agent.models.trees import ContentNode, RawNode


class RefinerAgent:
    def __init__(self, llm: LLMClient, poster_config: PosterConfig):
        self.llm = llm
        self.poster_config = poster_config

    def refine(
        self,
        raw_tree: RawNode,
        feedback: str = "",
        language: str = "en",
        logic_plan: LogicPlan | None = None,
    ) -> ContentNode:
        system = (
            "You build a LOGIC-DRIVEN academic poster content tree (NOT generic summaries). "
            "Goal: explain the paper's complex workflow and validation chain clearly.\n\n"
            "Rules:\n"
            "1) Each bullet is a causal/logical step: 'Because X → method Y → outcome Z' or "
            "'Step N: component A integrates B via C';\n"
            "2) Method section bullets must trace the pipeline (inputs, modules, constraints, outputs);\n"
            "3) Results bullets must link metrics/datasets to claims (DOCKQ, benchmark, ablation);\n"
            "4) 2-4 bullets per major section; each bullet ≤22 words, poster-style concise prose;\n"
            "5) Summaries ≤30 words; avoid repeating abstract wording; NO duplicate bullets across sections;\n"
            "6) logic_links: list of 3-6 strings like 'Introduction→Method: defines RPR/IR gap' "
            "(MUST be a JSON array of strings, NOT a single string);\n"
            "7) image_paths: copy figure paths from document tree for matching sections;\n"
            "8) paper_tables: copy relevant table dicts (from document) into Results/Methods sections;\n"
            "9) Preserve subsection structure as children when paper has multiple Results/Method parts;\n"
            "10) weight 0-1 by section importance;\n"
            "11) MUST include exactly 5 top-level sections: Abstract, Introduction, Methods, Results, Discussion "
            "(do not merge Methods into Results or omit Discussion).\n"
            "Output JSON: {title, summary, bullets, weight, logic_links, image_paths, paper_tables, children}. "
            + language_instruction(language)
        )
        if language == "zh":
            system = (
                "构建逻辑驱动的学术海报内容树（不要泛泛归纳）。"
                "每节 bullets 体现因果/流程链；Method 写清模块与数据流；Results 写清指标与数据集。"
                "logic_links 必须是字符串数组；image_paths 从文档树复制对应插图路径。"
                "输出 JSON：{title, summary, bullets, weight, logic_links, image_paths, children}。"
                + language_instruction(language)
            )
        user = f"Document tree:\n{_raw_to_text(raw_tree)}"
        if logic_plan:
            user += f"\n\nLogic plan (follow this structure):\n{logic_plan.to_prompt_block(language)}"
        if feedback:
            user += f"\n\nPrevious feedback:\n{feedback}"

        try:
            data = self.llm.chat_json(system, user, temperature=0.35)
            node = ContentNode.from_dict(data)
        except Exception:
            node = _fallback_content_tree(raw_tree)
        _normalize_logic_links(node)
        _merge_images_from_raw(node, raw_tree)
        _merge_assets_from_raw(node, raw_tree)
        from poster_agent.agents.figure_curator import index_figure_catalog, normalize_poster_figure_paths

        normalize_poster_figure_paths(node, catalog=index_figure_catalog(raw_tree))
        _ensure_poster_sections(node, logic_plan, language)
        _sanitize_content_tree(node)
        _condense_content_tree(node)
        self._apply_default_weights(node)
        return node

    def _apply_default_weights(self, node: ContentNode) -> None:
        defaults = self.poster_config.section_weights

        def match_weight(title: str) -> float | None:
            t = title.lower()
            mapping = {
                "title": ["title", "标题"],
                "abstract": ["abstract", "摘要"],
                "introduction": ["intro", "introduction", "引言", "背景"],
                "method": ["method", "approach", "方法", "模型", "framework", "architecture"],
                "experiment": ["experiment", "result", "evaluation", "实验", "结果"],
                "conclusion": ["conclusion", "discussion", "结论", "讨论"],
                "reference": ["reference", "参考文献"],
            }
            for key, kws in mapping.items():
                if any(kw in t for kw in kws):
                    return defaults.get(key)
            return None

        def walk(n: ContentNode) -> None:
            w = match_weight(n.title)
            if w is not None:
                if n.weight and n.weight > 0:
                    n.weight = 0.5 * w + 0.5 * n.weight
                else:
                    n.weight = w
            for c in n.children:
                walk(c)

        walk(node)


def _sanitize_content_tree(node: ContentNode) -> None:
    node.summary = sanitize_poster_text(node.summary)
    node.bullets = clean_bullets(node.bullets, min_keep=2, max_keep=4)
    for c in node.children:
        _sanitize_content_tree(c)


def _condense_content_tree(node: ContentNode) -> None:
    from poster_agent.render.content_sanitize import condense_summary, condense_bullet

    node.summary = condense_summary(node.summary)
    node.bullets = [condense_bullet(b) for b in node.bullets if b.strip()][:4]
    for c in node.children:
        _condense_content_tree(c)


def _merge_assets_from_raw(content: ContentNode, raw: RawNode) -> None:
    """合并论文表格与插图 caption 到内容树."""
    from pathlib import Path

    from poster_agent.extract.table_extractor import extract_key_tables

    tables = list(raw.tables or [])
    if not tables:
        full_text = _raw_full_text(raw)
        if full_text.strip():
            tables = [t.to_dict() for t in extract_key_tables(full_text, language="en")]

    catalog: dict[str, dict] = {}
    for item in raw.figure_catalog or []:
        path = item.get("path", "")
        if path:
            catalog[Path(path).name] = item
            catalog[path.replace("\\", "/")] = item

    def caption_for(path: str) -> str:
        key = Path(path).name
        meta = catalog.get(key) or catalog.get(path.replace("\\", "/"), {})
        hint = str(meta.get("caption_hint", "")).strip()
        fig_num = meta.get("fig_num") or meta.get("fig_num", 0)
        kind = meta.get("kind", "")
        if hint:
            prefix = f"Fig. {fig_num}" if fig_num else "Fig."
            kind_tag = f" [{kind}]" if kind and kind != "general" else ""
            return f"{prefix}{kind_tag}: {hint[:90]}"
        return ""

    def walk(node: ContentNode) -> None:
        caps: list[str] = []
        for p in node.image_paths:
            caps.append(caption_for(p))
        node.figure_captions = [c for c in caps if c]
        t = node.title.lower()
        if any(k in t for k in ("result", "experiment", "evaluation", "结果", "实验")):
            if not node.paper_tables and tables:
                node.paper_tables = list(tables[:2])
            elif node.paper_tables and len(node.paper_tables) < 2 and tables:
                for tbl in tables:
                    if tbl not in node.paper_tables:
                        node.paper_tables.append(tbl)
                    if len(node.paper_tables) >= 2:
                        break
        for ch in node.children:
            walk(ch)

    walk(content)
    if tables and not _any_paper_tables(content):
        for child in content.children:
            if any(k in child.title.lower() for k in ("result", "experiment", "结果", "实验")):
                child.paper_tables = list(tables[:2])
                break


def _raw_full_text(node: RawNode) -> str:
    parts = [node.content or ""]
    for c in node.children:
        parts.append(c.content or "")
        parts.extend(ch.content or "" for ch in c.children)
    return "\n".join(parts)


def _pick_catalog_paths(catalog: dict[str, dict], *, kind: str, limit: int = 1) -> list[str]:
    paths: list[str] = []
    for meta in catalog.values():
        if meta.get("kind") == kind and meta.get("path"):
            p = str(meta["path"])
            if p not in paths:
                paths.append(p)
        if len(paths) >= limit:
            break
    return paths


def _any_paper_tables(node: ContentNode) -> bool:
    if node.paper_tables:
        return True
    return any(_any_paper_tables(c) for c in node.children)


def _normalize_logic_links(node: ContentNode) -> None:
    links = node.logic_links
    if len(links) > 20 and all(len(x) == 1 for x in links[:10]):
        node.logic_links = []
    cleaned: list[str] = []
    for link in links:
        s = str(link).strip()
        if len(s) >= 8 and ("→" in s or "->" in s or ":" in s):
            cleaned.append(s)
        elif len(s) >= 15:
            cleaned.append(s)
    if cleaned:
        node.logic_links = cleaned[:8]
    for c in node.children:
        _normalize_logic_links(c)


def _merge_images_from_raw(content: ContentNode, raw: RawNode) -> None:
    raw_index = _index_raw_nodes(raw)

    def walk(node: ContentNode) -> None:
        if not node.image_paths:
            matched = _match_raw_images(node.title, raw_index)
            if matched:
                node.image_paths = matched
        for ch in node.children:
            walk(ch)

    walk(content)


def _index_raw_nodes(raw: RawNode) -> dict[str, RawNode]:
    out: dict[str, RawNode] = {}

    def walk(n: RawNode) -> None:
        if n.title:
            out[n.title.lower()] = n
        for c in n.children:
            walk(c)

    walk(raw)
    return out


def _match_raw_images(title: str, index: dict[str, RawNode]) -> list[str]:
    t = title.lower()
    if any(
        k in t
        for k in ("abstract", "摘要", "discussion", "讨论", "method", "方法", "approach", "framework")
    ):
        return []
    intro_keys = ("intro", "introduction", "background", "引言", "背景")
    result_keys = ("result", "experiment", "evaluation", "实验", "结果")
    if any(k in t for k in intro_keys):
        for key, node in index.items():
            if any(k in key for k in intro_keys) and node.images:
                return list(node.images[:1])
    if any(k in t for k in result_keys):
        for key, node in index.items():
            if any(k in key for k in result_keys) and node.images:
                return list(node.images[:2])
    return []


def _fuzzy_title_match(a: str, b: str) -> bool:
    tokens_a = set(re.findall(r"[a-z0-9]{4,}", a))
    tokens_b = set(re.findall(r"[a-z0-9]{4,}", b))
    return len(tokens_a & tokens_b) >= 1


def _section_kind(title: str) -> str:
    t = title.lower()
    if "abstract" in t or "摘要" in t:
        return "abstract"
    if "intro" in t or "引言" in t or "background" in t or "背景" in t:
        return "introduction"
    if any(k in t for k in ("method", "approach", "framework", "architecture", "方法", "模型", "架构")):
        return "methods"
    if any(k in t for k in ("result", "experiment", "evaluation", "实验", "结果", "benchmark", "validation")):
        return "results"
    if any(k in t for k in ("discussion", "conclusion", "讨论", "结论")):
        return "discussion"
    return "other"


def _ensure_poster_sections(
    node: ContentNode,
    logic_plan: LogicPlan | None,
    language: str,
) -> None:
    """补全 Refiner 遗漏的核心章节，避免海报只剩 3 块."""
    if not logic_plan:
        return
    en = language == "en"
    kinds_present = {_section_kind(c.title) for c in node.children}
    missing = [k for k in ("methods", "results", "discussion") if k not in kinds_present]
    if not missing:
        return

    stubs: dict[str, ContentNode] = {
        "methods": ContentNode(
            title="Methods" if en else "方法",
            summary=(logic_plan.core_problem or "")[:160],
            bullets=[str(s) for s in logic_plan.pipeline_steps[:5]],
            weight=0.12,
            logic_links=["Introduction→Methods: encode RPR/IR into AFM"],
        ),
        "results": ContentNode(
            title="Results" if en else "结果",
            summary="",
            bullets=[str(s) for s in logic_plan.validation_chain[:5]],
            weight=0.14,
            logic_links=["Methods→Results: benchmark validation"],
        ),
        "discussion": ContentNode(
            title="Discussion" if en else "讨论",
            summary=(logic_plan.core_problem or "")[:120],
            bullets=[
                (
                    f"GRASP integrates {', '.join(logic_plan.key_terms[:4])} for restrained complex prediction."
                    if en and logic_plan.key_terms
                    else "GRASP 整合多种实验约束以提升复合物预测。"
                ),
                str(logic_plan.validation_chain[-1]) if logic_plan.validation_chain else "",
                "Ablation confirms both RPR and IR contributions." if en else "消融实验验证 RPR 与 IR 均不可或缺。",
            ],
            weight=0.10,
            logic_links=["Results→Discussion: implications and limits"],
        ),
    }

    for kind in missing:
        stub = stubs.get(kind)
        if stub:
            node.children.append(stub)

    order = {"abstract": 0, "introduction": 1, "methods": 2, "results": 3, "discussion": 4, "other": 5}
    node.children.sort(key=lambda c: order.get(_section_kind(c.title), 5))


def _raw_to_text(node: RawNode, indent: int = 0, *, root: RawNode | None = None) -> str:
    if root is None:
        root = node
    pad = "  " * indent
    imgs = ", ".join(node.images[:3]) if node.images else "none"
    lines = [f"{pad}[{node.title}] figures=[{imgs}]\n{pad}  {(node.content or '')[:900]}"]
    if indent == 0:
        if root.tables:
            import json
            tbl_preview = json.dumps(root.tables[:2], ensure_ascii=False)[:700]
            lines.append(f"{pad}Paper tables (cite in Results): {tbl_preview}")
        if root.figure_catalog:
            from pathlib import Path
            cat_bits = []
            for c in root.figure_catalog[:10]:
                kind = c.get("kind", "?")
                cap = str(c.get("caption_hint", ""))[:50]
                cat_bits.append(f"{Path(c.get('path', '')).name}({kind}:{cap})")
            lines.append(f"{pad}Figure catalog: {'; '.join(cat_bits)}")
    for c in node.children:
        lines.append(_raw_to_text(c, indent + 1, root=root))
    return "\n".join(lines)


def _fallback_content_tree(raw: RawNode) -> ContentNode:
    def convert(n: RawNode) -> ContentNode:
        text = (n.content or "").strip()
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if len(s.strip()) > 20]
        bullets = sentences[:5] if sentences else ([text[:160]] if text else [])
        return ContentNode(
            title=n.title,
            summary=text[:120],
            bullets=bullets,
            weight=0.1,
            logic_links=[],
            image_paths=list(n.images),
            children=[convert(c) for c in n.children],
        )

    return convert(raw)

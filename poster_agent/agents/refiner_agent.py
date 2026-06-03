"""精炼智能体：原始文档树 → 内容树（层级压缩 + 逻辑链保留）."""

from __future__ import annotations

from poster_agent.llm_client import LLMClient
from poster_agent.config import PosterConfig
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
    ) -> ContentNode:
        system = (
            "You are an academic poster content refiner. Compress the document tree for poster display. "
            "Requirements: 1) Keep cross-section logic links; 2) 2-4 bullets per section; "
            "3) summary within 120 chars; 4) weight 0-1 for section importance. "
            "Output JSON: {title, summary, bullets, weight, logic_links, children}. "
            + language_instruction(language)
        )
        if language == "zh":
            system = (
                "你是学术海报内容精炼专家。将文档树压缩为适合海报展示的内容树。"
                "要求：1) 保留章节间逻辑依赖（logic_links）；2) 每节 2-4 条 bullets；"
                "3) summary 不超过 80 字；4) weight 表示章节重要性。"
                "输出 JSON：{title, summary, bullets, weight, logic_links, children}。"
                + language_instruction(language)
            )
        user = f"Document tree:\n{_raw_to_text(raw_tree)}"
        if feedback:
            user += f"\n\nPrevious feedback (please improve accordingly):\n{feedback}"

        try:
            data = self.llm.chat_json(system, user)
            node = ContentNode.from_dict(data)
        except Exception:
            node = _fallback_content_tree(raw_tree)
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
                "method": ["method", "approach", "方法", "模型"],
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
                n.weight = w
            for c in n.children:
                walk(c)

        walk(node)


def _raw_to_text(node: RawNode, indent: int = 0) -> str:
    pad = "  " * indent
    lines = [f"{pad}[{node.title}] {node.content[:500]}"]
    for c in node.children:
        lines.append(_raw_to_text(c, indent + 1))
    return "\n".join(lines)


def _fallback_content_tree(raw: RawNode) -> ContentNode:
    def convert(n: RawNode) -> ContentNode:
        text = n.content.strip()
        bullets = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()][:4]
        if not bullets and text:
            bullets = [text[:120]]
        return ContentNode(
            title=n.title,
            summary=text[:80],
            bullets=bullets,
            weight=0.1,
            logic_links=[],
            children=[convert(c) for c in n.children],
        )

    return convert(raw)

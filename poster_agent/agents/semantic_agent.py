"""语义检查智能体：跨节点逻辑一致性检查."""

from __future__ import annotations

from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode
from poster_agent.render.language import language_instruction


class SemanticCheckAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def check(self, content_tree: ContentNode, language: str = "en") -> tuple[float, list[str]]:
        system = (
            "You are an academic logic reviewer. Check whether the poster content tree explains "
            "the paper's WORKFLOW and VALIDATION chain (not just keyword summaries). "
            "Focus: causal links between sections; method steps match results metrics; "
            "introduction problem is addressed in method/results. "
            "Output JSON: {score: 0-1, issues: [descriptions], suggestions: [improvements]}. "
            + language_instruction(language)
        )
        if language == "zh":
            system = (
                "检查海报内容树是否清晰解释了论文的方法流程与验证链（而非关键词堆砌）。"
                "重点：章节间因果衔接；方法步骤与结果指标对应；引言问题在方法/结果中得到回应。"
                "输出 JSON：{score, issues, suggestions}。"
                + language_instruction(language)
            )
        user = (
            f"Content tree:\n{_content_to_text(content_tree, language)}"
            if language == "en"
            else f"内容树：\n{_content_to_text(content_tree, language)}"
        )
        result = self.llm.chat_json(system, user)
        score = float(result.get("score", 0.7))
        issues = result.get("issues") or []
        suggestions = result.get("suggestions") or []
        if not isinstance(issues, list):
            issues = [str(issues)]
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)]
        if language == "en":
            messages = [f"[logic] {i}" for i in issues] + [f"[suggest] {s}" for s in suggestions]
        else:
            messages = [f"[逻辑] {i}" for i in issues] + [f"[建议] {s}" for s in suggestions]
        return score, messages


def _content_to_text(node: ContentNode, language: str = "en", indent: int = 0) -> str:
    pad = "  " * indent
    none_label = "none" if language == "en" else "无"
    links = ", ".join(str(x) for x in node.logic_links) if node.logic_links else none_label
    lines = [
        f"{pad}[{node.title}] weight={node.weight:.2f}",
        f"{pad}  summary: {node.summary}",
        f"{pad}  bullets: {'; '.join(node.bullets)}",
        f"{pad}  logic_links: {links}",
    ]
    for c in node.children:
        lines.append(_content_to_text(c, language, indent + 1))
    return "\n".join(lines)

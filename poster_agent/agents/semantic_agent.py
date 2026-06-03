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
            "You are an academic logic reviewer. Check cross-section semantic consistency of the poster content tree. "
            "Focus: whether experiments match methods; whether conclusions support experiments; whether introduction questions are addressed. "
            "Output JSON: {score: 0-1, issues: [descriptions], suggestions: [improvements]}. "
            + language_instruction(language)
        )
        if language == "zh":
            system = (
                "你是学术论文逻辑审查专家。检查海报内容树的跨章节语义一致性。"
                "重点：实验结果是否对应方法；结论是否支撑实验；引言问题是否在方法中回应。"
                "输出 JSON：{score: 0-1, issues: [具体问题描述], suggestions: [改进建议]}。"
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
        f"{pad}  summary: {node.summary}",
        f"{pad}  bullets: {'; '.join(node.bullets)}",
    ]
    for c in node.children:
        lines.append(_content_to_text(c, language, indent + 1))
    return "\n".join(lines)

"""多模态评论智能体：评估图文匹配度与视觉层次."""

from __future__ import annotations

from pathlib import Path

from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import PosterNode
from poster_agent.render.language import language_instruction


class CommenterAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(
        self,
        poster_tree: PosterNode,
        image_path: Path | None = None,
        balance_metrics: dict[str, float] | None = None,
        language: str = "en",
        render_issues: str = "",
    ) -> tuple[float, float, str]:
        layout_desc = _describe_layout(poster_tree)
        metrics_text = ""
        if balance_metrics:
            if language == "zh":
                metrics_text = (
                    f"\n布局指标：对齐={balance_metrics.get('alignment')}, "
                    f"留白={balance_metrics.get('whitespace')}, "
                    f"密度={balance_metrics.get('density')}, "
                    f"溢出惩罚={balance_metrics.get('overflow')}, "
                    f"稀疏区块={balance_metrics.get('sparse_sections', 0)}"
                )
            else:
                metrics_text = (
                    f"\nLayout metrics: alignment={balance_metrics.get('alignment')}, "
                    f"whitespace={balance_metrics.get('whitespace')}, "
                    f"density={balance_metrics.get('density')}, "
                    f"overflow_penalty={balance_metrics.get('overflow')}, "
                    f"sparse_sections={balance_metrics.get('sparse_sections', 0)}"
                )

        render_text = ""
        if render_issues:
            render_text = (
                f"\n\nRender diagnostics (Paper2Poster Commenter):\n{render_issues}"
                if language == "en"
                else f"\n\n渲染诊断（Paper2Poster Commenter）：\n{render_issues}"
            )

        system = (
            "You are an academic poster reviewer. From layout description and metrics, evaluate: "
            "1) semantic_completeness (0-1) whether content covers key paper points; "
            "2) image_text_match (0-1) whether block titles match content and hierarchy is clear. "
            "Output JSON: {semantic_completeness, image_text_match, feedback}. "
            + language_instruction(language)
        )
        if language == "zh":
            system = (
                "你是学术海报评审专家。根据布局描述和指标，评估："
                "1) semantic_completeness (0-1) 内容是否完整覆盖论文要点；"
                "2) image_text_match (0-1) 各区块标题与内容是否语义匹配、层次是否清晰。"
                "输出 JSON：{semantic_completeness, image_text_match, feedback}。"
                + language_instruction(language)
            )
        user = (
            f"Poster layout:\n{layout_desc}{metrics_text}{render_text}"
            if language == "en"
            else f"海报布局：\n{layout_desc}{metrics_text}{render_text}"
        )
        if image_path and image_path.exists():
            user += (
                f"\nRendered poster file: {image_path.name}"
                if language == "en"
                else f"\n已渲染海报文件：{image_path.name}"
            )

        result = self.llm.chat_json(system, user)
        sc = float(result.get("semantic_completeness", 0.7))
        itm = float(result.get("image_text_match", 0.7))
        feedback = result.get("feedback", "")
        return sc, itm, feedback


def _describe_layout(node: PosterNode, indent: int = 0) -> str:
    pad = "  " * indent
    r = node.rect
    lines = [
        f"{pad}[{node.title}] ({r.x},{r.y},{r.width}x{r.height})",
        f"{pad}  summary: {node.summary[:100]}",
        f"{pad}  bullets: {len(node.bullets)} items",
    ]
    for c in node.children:
        lines.append(_describe_layout(c, indent + 1))
    return "\n".join(lines)

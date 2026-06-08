"""逻辑规划智能体：复杂论文的方法链/验证链梳理（PosterForest 风格）."""

from __future__ import annotations

from dataclasses import dataclass, field

from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import RawNode
from poster_agent.render.language import language_instruction


@dataclass
class LogicPlan:
    core_problem: str = ""
    pipeline_steps: list[str] = field(default_factory=list)
    validation_chain: list[str] = field(default_factory=list)
    key_terms: list[str] = field(default_factory=list)
    poster_sections: list[str] = field(default_factory=list)

    def to_prompt_block(self, language: str = "en") -> str:
        if language == "zh":
            lines = [
                f"核心问题：{self.core_problem}",
                "方法/流程链：" + " → ".join(self.pipeline_steps[:6]),
                "验证链：" + " → ".join(self.validation_chain[:5]),
                "关键术语：" + ", ".join(self.key_terms[:12]),
                "建议海报章节：" + ", ".join(self.poster_sections[:8]),
            ]
        else:
            lines = [
                f"Core problem: {self.core_problem}",
                "Method/pipeline chain: " + " → ".join(self.pipeline_steps[:6]),
                "Validation chain: " + " → ".join(self.validation_chain[:5]),
                "Key terms: " + ", ".join(self.key_terms[:12]),
                "Suggested poster sections: " + ", ".join(self.poster_sections[:8]),
            ]
        return "\n".join(lines)


class LogicPlannerAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, raw_tree: RawNode, language: str = "en") -> LogicPlan:
        system = (
            "You analyze a complex scientific paper BEFORE poster summarization. "
            "Do NOT write generic summaries. Extract the LOGICAL FLOW:\n"
            "1) core_problem — what gap/limitation motivates the work\n"
            "2) pipeline_steps — ordered method stages (input → module → output), 4-7 steps\n"
            "3) validation_chain — how claims are verified (dataset → metric → ablation), 3-5 steps\n"
            "4) key_terms — domain-specific acronyms/components (≤12)\n"
            "5) poster_sections — 5-8 section titles preserving method/results structure\n"
            "Output JSON with those fields as lists/strings. "
            + language_instruction(language)
        )
        if language == "zh":
            system = (
                "在海报压缩前分析复杂论文的逻辑结构，不要写泛泛摘要。"
                "提取：core_problem、pipeline_steps（4-7步流程链）、"
                "validation_chain（3-5步验证链）、key_terms、poster_sections。"
                "输出 JSON。"
                + language_instruction(language)
            )
        user = f"Paper document tree:\n{_raw_to_text(raw_tree)}"
        try:
            data = self.llm.chat_json(system, user, temperature=0.25)
            return LogicPlan(
                core_problem=str(data.get("core_problem", "")),
                pipeline_steps=_as_str_list(data.get("pipeline_steps")),
                validation_chain=_as_str_list(data.get("validation_chain")),
                key_terms=_as_str_list(data.get("key_terms")),
                poster_sections=_as_str_list(data.get("poster_sections")),
            )
        except Exception:
            return _fallback_plan(raw_tree, language)


def _as_str_list(val) -> list[str]:
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str) and val.strip():
        return [val.strip()]
    return []


def _fallback_plan(raw: RawNode, language: str) -> LogicPlan:
    en = language == "en"
    sections = [c.title for c in raw.children if c.title]
    return LogicPlan(
        core_problem=raw.content[:200] if raw.content else ("Research problem" if en else "研究问题"),
        pipeline_steps=["Parse inputs", "Core model", "Integrate constraints", "Predict structure"]
        if en
        else ["输入解析", "核心模型", "约束整合", "结构预测"],
        validation_chain=["Benchmark", "Ablation", "Case study"] if en else ["基准测试", "消融", "案例分析"],
        poster_sections=sections[:8] or (["Introduction", "Method", "Results", "Conclusion"] if en else ["引言", "方法", "结果", "结论"]),
    )


def _raw_to_text(node: RawNode, indent: int = 0) -> str:
    pad = "  " * indent
    imgs = f" [figures: {len(node.images)}]" if node.images else ""
    lines = [f"{pad}[{node.title}]{imgs} {(node.content or '')[:800]}"]
    for c in node.children:
        lines.append(_raw_to_text(c, indent + 1))
    return "\n".join(lines)

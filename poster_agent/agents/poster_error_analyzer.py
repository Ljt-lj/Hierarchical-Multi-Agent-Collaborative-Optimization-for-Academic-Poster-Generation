"""GenPilot / Paper2Poster 风格：海报渲染错误分析与反馈映射."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from poster_agent.models.render_report import PosterRenderReport
from poster_agent.models.trees import ContentNode, PosterNode


class IssueKind(str, Enum):
    OVERFLOW = "overflow"
    SPARSE = "sparse"
    MISSING_VISUAL = "missing_visual"
    LOW_DENSITY = "low_density"
    WHITESPACE = "whitespace"


@dataclass
class PosterIssue:
    kind: IssueKind
    section: str
    message: str
    action: str  # refiner | layout | visual


class PosterErrorAnalyzer:
    """阶段一：错误分析 + 细粒度映射（对应 GenPilot Stage-1 / Paper2Poster Commenter）."""

    def analyze(
        self,
        render_report: PosterRenderReport | None,
        balance_metrics: dict[str, float] | None,
        content_tree: ContentNode | None = None,
    ) -> list[PosterIssue]:
        issues: list[PosterIssue] = []
        if render_report:
            for sec in render_report.sections:
                if sec.overflow:
                    issues.append(
                        PosterIssue(
                            IssueKind.OVERFLOW,
                            sec.title,
                            f"Section '{sec.title}': only {sec.bullets_shown}/{sec.bullets_total} bullets visible (text clipped).",
                            "refiner",
                        )
                    )
                if sec.sparse:
                    issues.append(
                        PosterIssue(
                            IssueKind.SPARSE,
                            sec.title,
                            f"Section '{sec.title}': too much empty space; add 2-4 substantive bullets with numbers/results.",
                            "refiner",
                        )
                    )
                if sec.missing_visual:
                    issues.append(
                        PosterIssue(
                            IssueKind.MISSING_VISUAL,
                            sec.title,
                            f"Section '{sec.title}': no chart/figure; add stat_cards, bar_chart, or paper figure.",
                            "visual",
                        )
                    )
                if sec.body_font < 32:
                    issues.append(
                        PosterIssue(
                            IssueKind.LOW_DENSITY,
                            sec.title,
                            f"Section '{sec.title}': body font {sec.body_font}px may be too small for poster readability.",
                            "layout",
                        )
                    )

        if balance_metrics:
            ws = float(balance_metrics.get("whitespace", 1.0))
            if ws < 0.55:
                issues.append(
                    PosterIssue(
                        IssueKind.WHITESPACE,
                        "global",
                        "Poster has excessive whitespace; increase bullets per section and add visuals.",
                        "refiner",
                    )
                )

        if content_tree and not render_report:
            issues.extend(self._analyze_content_only(content_tree))
        return issues

    def _analyze_content_only(self, root: ContentNode) -> list[PosterIssue]:
        issues: list[PosterIssue] = []
        for node in _walk_sections(root):
            if len(node.bullets) < 2:
                issues.append(
                    PosterIssue(
                        IssueKind.SPARSE,
                        node.title,
                        f"Section '{node.title}' has fewer than 2 bullets before render.",
                        "refiner",
                    )
                )
            if not node.visuals and not node.image_paths:
                t = node.title.lower()
                if not any(k in t for k in ("reference", "ref", "参考", "abstract", "摘要")):
                    issues.append(
                        PosterIssue(
                            IssueKind.MISSING_VISUAL,
                            node.title,
                            f"Section '{node.title}' has no visual planned.",
                            "visual",
                        )
                    )
        return issues

    def format_feedback(self, issues: list[PosterIssue], language: str = "en") -> str:
        if not issues:
            return ""
        header = (
            "Render error analysis (fix before next iteration):"
            if language == "en"
            else "渲染错误分析（下一轮迭代前请修复）："
        )
        lines = [header]
        for issue in issues[:12]:
            tag = issue.kind.value.upper()
            lines.append(f"- [{tag}] {issue.message} → action: {issue.action}")
        return "\n".join(lines)

    def format_refiner_instructions(self, issues: list[PosterIssue], language: str = "en") -> str:
        refiner = [i for i in issues if i.action == "refiner"]
        if not refiner:
            return ""
        if language == "zh":
            lines = ["针对 Refiner 的修改要求："]
        else:
            lines = ["Refiner-specific fixes:"]
        for i in refiner:
            if i.kind == IssueKind.OVERFLOW:
                lines.append(f"- {i.section}: shorten bullets OR split content; max 3 lines per bullet.")
            elif i.kind == IssueKind.SPARSE:
                lines.append(
                    f"- {i.section}: add 2-4 bullets with key numbers, dataset names, or contributions."
                )
            else:
                lines.append(f"- {i.message}")
        return "\n".join(lines)


def _walk_sections(node: ContentNode) -> list[ContentNode]:
    if node.children:
        return list(node.children)
    return [node]

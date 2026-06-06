"""插图质量评估：类型匹配、去重、可读性，必要时优化 spec 后重生成."""

from __future__ import annotations

from pathlib import Path

from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode
from poster_agent.models.visuals import VisualSpec
from poster_agent.render.language import language_instruction
from poster_agent.agents.visual_agent import is_generic_steps
from poster_agent.render.visual_registry import (
    VisualRegistry,
    is_abstract_title,
    is_conclusion_title,
)


class VisualEvaluator:
    PASS_SCORE = 0.78
    MAX_ATTEMPTS = 3

    def __init__(self, llm: LLMClient | None):
        self.llm = llm

    def evaluate_spec(
        self,
        spec: VisualSpec,
        node: ContentNode,
        registry: VisualRegistry,
        language: str = "en",
    ) -> tuple[float, list[str]]:
        issues = list(registry.duplicate_issues(spec))
        issues.extend(self._rule_check(spec, node))
        llm_score, llm_issues = self._llm_review(spec, node, registry, language)
        issues.extend(llm_issues)
        rule_score = max(0.0, 1.0 - 0.22 * len(set(issues)))
        score = 0.45 * rule_score + 0.55 * llm_score
        return score, issues

    def evaluate_image(self, path: Path | None, spec: VisualSpec) -> list[str]:
        if not path or not path.exists():
            return ["image not generated"]
        issues: list[str] = []
        if spec.type == "stat_cards":
            issues.extend(_check_stat_cards_layout(path))
        try:
            from PIL import Image
            img = Image.open(path)
            w, h = img.size
            if w < 80 or h < 40:
                issues.append("image too small")
        except Exception as exc:
            issues.append(f"image unreadable: {exc}")
        return issues

    def refine_spec(
        self,
        spec: VisualSpec,
        node: ContentNode,
        issues: list[str],
        registry: VisualRegistry,
        language: str = "en",
    ) -> VisualSpec:
        if spec.type == "stat_cards" and is_abstract_title(node.title):
            return self._rule_refine(spec, node, issues, language)
        if self.llm:
            refined = self._llm_refine(spec, node, issues, registry.summary(), language)
            if refined:
                if is_abstract_title(node.title):
                    refined = self._restore_abstract_metrics(refined, spec)
                return refined
        return self._rule_refine(spec, node, issues, language)

    def _restore_abstract_metrics(self, refined: VisualSpec, original: VisualSpec) -> VisualSpec:
        """评估优化不得把 Abstract 数值指标卡改成文字卡片."""
        orig_cards = (original.data or {}).get("cards") or []
        if not orig_cards:
            return refined
        if refined.type != "stat_cards":
            return VisualSpec(
                type="stat_cards",
                title=original.title,
                section_title=original.section_title,
                data={"cards": orig_cards[:3]},
            )
        cards = (refined.data or {}).get("cards") or []
        if not cards or not _cards_have_numeric_values(cards):
            return VisualSpec(
                type="stat_cards",
                title=original.title,
                section_title=original.section_title,
                data={"cards": orig_cards[:3]},
            )
        return refined

    def _rule_check(self, spec: VisualSpec, node: ContentNode) -> list[str]:
        issues: list[str] = []
        t = node.title.lower()
        if spec.type == "stat_cards" and not is_abstract_title(t):
            issues.append("wrong type for section")
        if is_conclusion_title(t) and spec.type in ("stat_cards", "bar_chart"):
            issues.append("conclusion should not repeat metric charts")
        cards = spec.data.get("cards") or []
        if spec.type == "stat_cards" and len(cards) > 3:
            issues.append("too many stat cards")
        for card in cards:
            if isinstance(card, dict) and len(str(card.get("label", ""))) > 16:
                issues.append("stat card label too long")
        labels = spec.data.get("labels") or []
        for lb in labels:
            if len(str(lb)) > 14:
                issues.append("chart label too long")
        steps = spec.data.get("steps") or []
        if spec.type in ("flow_diagram", "architecture") and is_generic_steps(steps):
            issues.append("generic placeholder steps")
        for step in steps:
            if len(str(step)) > 90:
                issues.append("flow step too long")
        return issues

    def _llm_review(
        self,
        spec: VisualSpec,
        node: ContentNode,
        registry: VisualRegistry,
        language: str,
    ) -> tuple[float, list[str]]:
        if not self.llm:
            return 0.85, []
        system = (
            "You review scientific poster visuals before rendering. "
            "Score 0-1 on: type-content fit, data extracted from section (not generic), "
            "no duplication with prior sections, label brevity, clarity. "
            + language_instruction(language)
            + ' Output JSON: {"score":0.85,"issues":["..."],"ok":true}'
        )
        user = (
            f"Section: {node.title}\nSummary: {node.summary}\nBullets: {node.bullets}\n"
            f"Visual spec: {spec.to_dict()}\nAlready used:\n{registry.summary()}"
        )
        try:
            result = self.llm.chat_json(system, user, temperature=0.2)
            if isinstance(result, dict):
                score = float(result.get("score", 0.75))
                issues = [str(x) for x in (result.get("issues") or []) if x]
                return max(0.0, min(1.0, score)), issues
        except Exception:
            pass
        return 0.75, []

    def _llm_refine(
        self,
        spec: VisualSpec,
        node: ContentNode,
        issues: list[str],
        used_summary: str,
        language: str,
    ) -> VisualSpec | None:
        system = (
            "Fix a poster visual specification. Use data ONLY from the section text. "
            "Short labels (<=12 chars). Unique vs already_used. "
            "stat_cards only for Abstract with max 3 cards. "
            + language_instruction(language)
            + ' Output JSON: {"type","title","section_title","data"}'
        )
        user = (
            f"Section: {node.title}\n{node.summary}\n{node.bullets}\n"
            f"Current: {spec.to_dict()}\nIssues: {issues}\nAlready used:\n{used_summary}"
        )
        try:
            result = self.llm.chat_json(system, user, temperature=0.3)
            if isinstance(result, dict) and result.get("type"):
                result["section_title"] = node.title
                return VisualSpec.from_dict(result)
        except Exception:
            pass
        return None

    def _rule_refine(
        self,
        spec: VisualSpec,
        node: ContentNode,
        issues: list[str],
        language: str,
    ) -> VisualSpec:
        en = language == "en"
        data = dict(spec.data or {})
        if spec.type == "stat_cards":
            cards = []
            for c in data.get("cards") or []:
                if not isinstance(c, dict):
                    continue
                cards.append({
                    "label": str(c.get("label", "")),
                    "value": str(c.get("value", "")),
                    "unit": str(c.get("unit", "")),
                })
            data["cards"] = cards[:3]
        if spec.type in ("bar_chart", "line_chart"):
            labels = [_short(str(x), 12) for x in (data.get("labels") or [])]
            data["labels"] = labels
        if spec.type in ("flow_diagram", "architecture"):
            steps = spec.data.get("steps") or []
            if is_generic_steps(steps):
                steps = node.bullets[:4] or [node.summary or node.title]
            data["steps"] = [_short(str(s), 72) for s in steps[:4]]
        if "wrong type" in " ".join(issues).lower() or "conclusion" in " ".join(issues).lower():
            if is_conclusion_title(node.title):
                return VisualSpec(
                    type="bullet_cards",
                    title="Key Takeaways" if en else "要点总结",
                    section_title=node.title,
                    data={"items": node.bullets[:3] or [node.summary or ""]},
                )
        return VisualSpec(type=spec.type, title=spec.title, section_title=node.title, data=data)


def _short(text: str, n: int) -> str:
    text = text.strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _cards_have_numeric_values(cards: list) -> bool:
    import re
    for c in cards:
        if not isinstance(c, dict):
            continue
        val = str(c.get("value", ""))
        if re.search(r"\d", val):
            return True
    return False


def _check_stat_cards_layout(path: Path) -> list[str]:
    """启发式：指标卡区域非空白且尺寸合理."""
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if h < 120:
            return ["stat cards height too small"]
        px = img.load()
        dark = 0
        for y in range(h):
            for x in range(w):
                r, g, b = px[x, y] if not isinstance(px[x, y], tuple) else px[x, y][:3]
                if r < 240 or g < 240 or b < 240:
                    dark += 1
        if dark < w * h * 0.02:
            return ["stat cards appear empty"]
    except Exception:
        pass
    return []

"""全局控制智能体：融合评分，决定迭代终止."""

from __future__ import annotations

from poster_agent.config import PosterConfig
from poster_agent.models.trees import EvaluationScore


class ControllerAgent:
    def __init__(self, config: PosterConfig):
        self.config = config

    def decide(
        self,
        semantic_score: float,
        layout_balance: float,
        image_text_match: float,
        logic_score: float,
        iteration: int,
        feedback: str,
    ) -> tuple[EvaluationScore, bool]:
        semantic_completeness = 0.6 * semantic_score + 0.4 * logic_score
        score = EvaluationScore(
            semantic_completeness=semantic_completeness,
            layout_balance=layout_balance,
            image_text_match=image_text_match,
            iteration_count=iteration,
            feedback=feedback,
        )
        should_stop = (
            score.overall >= self.config.score_threshold
            or iteration >= self.config.max_iterations
        )
        return score, should_stop

    def build_refiner_feedback(
        self,
        score: EvaluationScore,
        logic_issues: list[str],
        *,
        error_feedback: str = "",
        refiner_instructions: str = "",
    ) -> str:
        parts = [score.feedback] if score.feedback else []
        if error_feedback:
            parts.append(error_feedback)
        if refiner_instructions:
            parts.append(refiner_instructions)
        parts.extend(logic_issues)
        parts.append(
            f"当前综合评分 {score.overall:.2f}，目标 {self.config.score_threshold}。"
            f"语义={score.semantic_completeness:.2f}, "
            f"布局={score.layout_balance:.2f}, "
            f"图文={score.image_text_match:.2f}"
        )
        return "\n".join(parts)
